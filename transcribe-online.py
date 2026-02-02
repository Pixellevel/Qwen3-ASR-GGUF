
import os
import sys
import time
import codecs
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

# ==========================================
# 1. 协议定义 (Dataclasses)
# ==========================================
class MsgType(Enum):
    CMD_ENCODE = auto()   # 主进程 -> Encoder: 编码请求
    CMD_STOP = auto()     # 主进程 -> Encoder: 停止请求
    MSG_EMBD = auto()     # Encoder -> 主进程: 返回特征
    MSG_READY = auto()    # Encoder -> 主进程: 就绪信号
    MSG_DONE = auto()     # Encoder -> 主进程: 已退出信号

@dataclass
class StreamingMessage:
    msg_type: MsgType
    data: Any = None      # 存放音频 chunk 或 embedding 结果
    is_last: bool = False # 标记是否为最后一段音频
    encode_time: float = 0.0 # 编码器实际耗时

# ==========================================
# 核心配置参数 (User Control)
# ==========================================
CHUNK_DURATION = 40.0     # 音频切片时长 (秒)
MEMORY_VAR_COUNT = 2      # 记忆中保留的音频片段数量
ROLLBACK_VAR_COUNT = 5    # 回滚/撤销的 Token 数量 (不显示的延迟缓冲区大小)

# 固定路径配置
PROJECT_ROOT = Path(__file__).parent.absolute()
FRONTEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_frontend.onnx")
BACKEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_backend.int8.onnx")
LLM_GGUF_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_asr_llm.q8_0.gguf")

# ==========================================
# 2. 编码器进程 (Encoder Worker & Preprocessor)
# ==========================================
class FastWhisperMel:
    """完全基于 NumPy 和 Librosa 的 Mel 提取器 (替代 Transformers)"""
    def __init__(self, filter_path):
        self.filters = np.load(filter_path) # (201, 128)
        
    def __call__(self, audio):
        import librosa
        # 1. STFT (Reflect padding, Hann window)
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
        # 2. Power Spectrum
        magnitudes = np.abs(stft) ** 2
        # 3. Mel Filterbank ( official filters are (201, 128) )
        mel_spec = np.dot(self.filters.T, magnitudes)
        # 4. Log Mel
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        # 5. Normalization
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

def encoder_worker_proc(to_enc_q, from_enc_q):
    """独立进程运行的编码器：按需编码，瞬间启动"""
    import onnxruntime as ort
    
    # 路径定义
    MEL_FILTERS_PATH = os.path.join(PROJECT_ROOT, "model", "mel_filters.npy")
    
    # 初始化
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    
    frontend_sess = ort.InferenceSession(FRONTEND_ONNX_PATH, sess_options=sess_opts, providers=providers)
    backend_sess = ort.InferenceSession(BACKEND_ONNX_PATH, sess_options=sess_opts, providers=providers)
    
    mel_extractor = FastWhisperMel(MEL_FILTERS_PATH)
    
    # GPU Warmup: 跑一段随机音频以触发 Shader 编译和显存分配
    # 这将 1-2 秒的“冷启动”耗时提前到初始化阶段
    dummy_wav = np.random.randn(int(16000 * 2)).astype(np.float32) # 2秒随机噪声
    dummy_mel = mel_extractor(dummy_wav)
    dummy_input = dummy_mel.T[np.newaxis, ...].astype(np.float32)
    feat_out = frontend_sess.run(None, {"mel": dummy_input})[0]
    _ = backend_sess.run(None, {"feat_in": feat_out})[0]
    
    # 发送就绪信号
    from_enc_q.put(StreamingMessage(MsgType.MSG_READY))
    
    while True:
        msg: StreamingMessage = to_enc_q.get()
        
        if msg.msg_type == MsgType.CMD_STOP:
            from_enc_q.put(StreamingMessage(MsgType.MSG_DONE))
            break
            
        if msg.msg_type == MsgType.CMD_ENCODE:
            audio_chunk = msg.data
            t0 = time.time()
            # 执行简易 Mel 提取
            mel = mel_extractor(audio_chunk) # 返回 (128, T)
            # ONNX 前端需要 (Batch, Seq, 128)
            mel_input = mel.T[np.newaxis, ...].astype(np.float32)
            
            # 推理
            feat_out = frontend_sess.run(None, {"mel": mel_input})[0]
            audio_embd = backend_sess.run(None, {"feat_in": feat_out})[0]
            if audio_embd.ndim == 3: audio_embd = audio_embd[0]
            t_encode = time.time() - t0
            
            from_enc_q.put(StreamingMessage(MsgType.MSG_EMBD, data=audio_embd, is_last=msg.is_last, encode_time=t_encode))

# ==========================================
# 3. 核心流式器 (Master)
# ==========================================
class ChunkSegment:
    def __init__(self, audio_embd):
        self.audio_embd = audio_embd
        self.committed_text = "" # 该片段锁定的稳定文本

class InfiniteStreamer31:
    def __init__(self, context=""):
        import multiprocessing as mp
        # from tokenizers import Tokenizer
        t_start = time.time()
        print(f"--- 初始化 Infinite Streaming Engine (Bidirectional Handshake & Native GGUF) ---")
        
        self.context = context
        # TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "tokenizer.json")
        # self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
        # 延迟导入 LLM 组件
        from qwen_asr_gguf import llama
        self.model = llama.LlamaModel(LLM_GGUF_PATH)
        self.embedding_table = llama.get_token_embeddings_gguf(LLM_GGUF_PATH)
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, n_batch=4096, embeddings=False)
        self.sampler = llama.LlamaSampler(temperature=0.4)
        
        # 建立消息队列
        self.to_enc_q = mp.Queue()
        self.from_enc_q = mp.Queue()
        
        # 启动编码器进程
        self.enc_proc = mp.Process(target=encoder_worker_proc, args=(self.to_enc_q, self.from_enc_q), daemon=True)
        self.enc_proc.start()
        
        # 等待就绪
        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_READY:
            print("Encoder Worker Ready.")

        # 状态管理
        self.segment_queue = deque(maxlen=MEMORY_VAR_COUNT)
        self.archive_text = ""
        
        # 统计数据
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.total_encode_time = 0.0
        self.total_wait_time = 0.0
        self.load_time = time.time() - t_start
        
        # 基础 Token ID (Native Lookup)
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def decode_tokens(self, tokens):
        if not tokens: return ""
        # 统一使用 native detokenize
        return self.model.detokenize(tokens)

    def shutdown(self):
        self.to_enc_q.put(StreamingMessage(MsgType.CMD_STOP))
        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_DONE:
            print("\n\nEncoder Worker Terminated Safely.")
        self.enc_proc.join()

    def run_llm_buffered(self, audio_embd, prefix_text, is_last_chunk=False, language: str = None):
        import numpy as np
        import codecs
        from qwen_asr_gguf import llama
        
        # 1. Prompt Construction
        system_text = "You are a helpful assistant. "
        user_prompt_text = f"{self.context}\n\n" if self.context else ""

        def tk(t): return self.model.tokenize(t)

        prefix_tokens = [self.ID_IM_START] + tk(f"system\n{system_text}") + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(f"user\n{user_prompt_text}") + [self.ID_AUDIO_START]
        
        # 构建 Assistant 引导部分
        assistant_prompt = "assistant\n"
        if language:
            assistant_prompt += f"language {language}"

        suffix_tokens = [self.ID_AUDIO_END] + tk("语音转录：") + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(assistant_prompt) + [self.ID_ASR_TEXT] + \
                        tk(prefix_text)

        n_prefix = len(prefix_tokens)
        n_audio = audio_embd.shape[0]
        n_suffix = len(suffix_tokens)
        total_len = n_prefix + n_audio + n_suffix
        
        full_embd = np.zeros((total_len, self.model.n_embd), dtype=np.float32)
        full_embd[:n_prefix] = self.embedding_table[prefix_tokens]
        full_embd[n_prefix : n_prefix + n_audio] = audio_embd
        full_embd[n_prefix + n_audio : n_prefix + n_audio + n_suffix] = self.embedding_table[suffix_tokens]
        
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        self.total_prefill_time += (time.time() - t_pre_start)
        self.total_prefill_tokens += total_len
        
        # 2. Generation Loop with Async Overlap
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        cur_pos = total_len
        gen_batch = llama.LlamaBatch(4, 0, 1)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        
        last_sampled_token = self.sampler.sample(self.ctx.ptr)
        
        for _ in range(150):
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            
            gen_batch.set_token(last_sampled_token, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            self.ctx.decode(gen_batch)
            
            # --- OVERLAP GAP: CPU Word ---
            display_queue.append(last_sampled_token)
            if len(display_queue) > ROLLBACK_VAR_COUNT:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    print(piece, end="", flush=True)
                    stable_text_acc += piece
            # ------------------------------
            
            cur_pos += 1
            last_sampled_token = self.sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1
            
        self.total_decode_time += (time.time() - t_gen_start)
        self.total_decode_tokens += n_gen_tokens
            
        if is_last_chunk:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    print(piece, end="", flush=True)
                    stable_text_acc += piece
            final_p = decoder.decode(b"", final=True)
            if final_p:
                print(final_p, end="", flush=True)
                stable_text_acc += final_p
            
        return prefix_text + stable_text_acc, stable_tokens

# ==========================================
# 4. 主程序 (Runner)
# ==========================================
def main():
    import numpy as np
    import librosa
    
    AUDIO_FILE = "test.mp3"
    CONTEXT = "这是1004期睡前消息，主持人叫督工，助理叫静静，"
    
    streamer = InfiniteStreamer31(context=CONTEXT)
    
    print(f"Loading audio: {AUDIO_FILE}")
    full_audio, sr = librosa.load(AUDIO_FILE, sr=16000)
    SAMPLES_PER_CHUNK = int(CHUNK_DURATION * sr)
    total_len = len(full_audio)
    num_chunks = int(np.ceil(total_len / SAMPLES_PER_CHUNK))
    
    print("--- Start Pipelined Streaming ---")
    t_main_start = time.time()
    
    # --- 启动第一个分块的编码 ---
    def get_chunk(idx):
        s = idx * SAMPLES_PER_CHUNK
        e = min((idx+1) * SAMPLES_PER_CHUNK, total_len)
        chunk = full_audio[s:e]
        if len(chunk) < SAMPLES_PER_CHUNK:
            chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))
        return chunk, (idx == num_chunks - 1)

    # 1. 发送第一个块
    chunk, is_last = get_chunk(0)
    streamer.to_enc_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=chunk, is_last=is_last))
    
    for i in range(num_chunks):
        # 2. 等待当前块的 Embedding
        t_w_start = time.time()
        msg: StreamingMessage = streamer.from_enc_q.get()
        streamer.total_wait_time += (time.time() - t_w_start)
        streamer.total_encode_time += msg.encode_time
        
        current_embd = msg.data
        was_last = msg.is_last
        
        # 3. 握手触发：立刻发送下一块的编码指令（如果有）
        if not was_last:
            next_chunk, next_is_last = get_chunk(i + 1)
            streamer.to_enc_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=next_chunk, is_last=next_is_last))
        
        # 4. 同时进行本块的 LLM 解码 (此时编码器正在后台算下一块)
        # 管理滑动窗口
        new_seg = ChunkSegment(current_embd)
        if len(streamer.segment_queue) >= MEMORY_VAR_COUNT:
            oldest = streamer.segment_queue.popleft()
            streamer.archive_text += oldest.committed_text
        streamer.segment_queue.append(new_seg)
        
        prefix_str = streamer.archive_text + "".join([s.committed_text for s in list(streamer.segment_queue)[:-1]])
        total_audio_input = np.concatenate([s.audio_embd for s in streamer.segment_queue], axis=0)
        
        full_out_text, _ = streamer.run_llm_buffered(total_audio_input, prefix_str, is_last_chunk=was_last)
        new_seg.committed_text = full_out_text[len(prefix_str):]

    t_total = time.time() - t_main_start
    streamer.shutdown()
    
    audio_duration = total_len / 16000
    rtf = t_total / audio_duration if audio_duration > 0 else 0
    
    print("\n\n--- Done ---")
    print(f"\n📊 性能统计 (Streaming Mode):")
    print(f"  🔹 初始化耗时: {streamer.load_time:.3f} s")
    print(f"  🔹 音频总时长: {audio_duration:.2f} s")
    print(f"  🔹 处理总耗时: {t_total:.3f} s (RTF: {rtf:.4f})")
    # print(f"  🔹 - 编码器总计耗时: {streamer.total_encode_time:.3f} s (Pipelined Overlapped)")
    print(f"  🔹 - 进入处理等待: {streamer.total_wait_time:.3f} s (主要为首段加载耗时)")
    print(f"  🔹 - LLM Prefill  : {streamer.total_prefill_time:.3f} s | {streamer.total_prefill_tokens} tokens | {streamer.total_prefill_tokens/streamer.total_prefill_time if streamer.total_prefill_time > 0 else 0:.1f} tokens/s")
    print(f"  🔹 - LLM Decode   : {streamer.total_decode_time:.3f} s | {streamer.total_decode_tokens} tokens | {streamer.total_decode_tokens/streamer.total_decode_time if streamer.total_decode_time > 0 else 0:.1f} tokens/s")
    # print(f"\n💡 注：由于采用了流水线并行，'编码器耗时' 与 'LLM耗时' 深度重叠，\n  实际感知卡顿和等待仅体现在 '进入处理等待'。")

if __name__ == "__main__":
    # Windows 环境多进程启动优化
    import warnings
    warnings.filterwarnings("ignore")
    main()
