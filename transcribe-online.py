
import os
import sys
import time
import codecs
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional
from qwen_asr_gguf import chinese_itn

# ==================== Vulkan 选项 ====================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "1"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）


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
# 2. 编码器进程 (Encoder Worker & Preprocessor)
# ==========================================
class FastWhisperMel:
    """完全基于 NumPy 和 Librosa 的 Mel 提取器 (替代 Transformers)"""
    def __init__(self, filter_path):
        self.filters = np.load(filter_path) # (201, 128)
        
    def __call__(self, audio, dtype=np.float32):
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
        return log_spec.astype(dtype)

def encoder_worker_proc(to_enc_q, from_enc_q, frontend_path, backend_path, mel_filters_path):
    """独立进程运行的编码器：按需编码，瞬间启动"""
    import onnxruntime as ort
    
    # 初始化
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CPUExecutionProvider']
    # if 'DmlExecutionProvider' in ort.get_available_providers():
    #     providers.insert(0, 'DmlExecutionProvider') 

    try:
        frontend_sess = ort.InferenceSession(frontend_path, sess_options=sess_opts, providers=providers)
        backend_sess = ort.InferenceSession(backend_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"[编码进程] 加载 ONNX 模型失败: {e}")
        # 发送错误信号或直接退出
        return
    
    mel_extractor = FastWhisperMel(mel_filters_path)
    
    # 检测模型输入类型 (Auto-detect FP32 vs FP16)
    try:
        fe_input_type = frontend_sess.get_inputs()[0].type # 'tensor(float)' or 'tensor(float16)'
        input_dtype = np.float16 if 'float16' in fe_input_type else np.float32
        # print(f"编码进程检测到输入精度: {input_dtype.__name__}")
    except:
        input_dtype = np.float32
        print(f"编码进程默认输入精度: float32")

    # GPU Warmup: 跑一段音频以触发 Shader 编译和显存分配
    warmup_seconds = 40
    dummy_wav = np.zeros(int(16000 * warmup_seconds), dtype=np.float32)
    try:
        dummy_mel = mel_extractor(dummy_wav, dtype=input_dtype)
        dummy_input = dummy_mel.T[np.newaxis, ...]
        feat_out = frontend_sess.run(None, {"mel": dummy_input})[0]
        _ = backend_sess.run(None, {"feat_in": feat_out})[0]
    except Exception as e:
        print(f"[编码进程] 预热失败 (已忽略): {e}")
    
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
            mel = mel_extractor(audio_chunk, dtype=input_dtype) # 返回 (128, T)
            # ONNX 前端需要 (Batch, Seq, 128)
            mel_input = mel.T[np.newaxis, ...]
            
            # 推理
            feat_out = frontend_sess.run(None, {"mel": mel_input})[0]
            audio_embd = backend_sess.run(None, {"feat_in": feat_out})[0]
            if audio_embd.ndim == 3: audio_embd = audio_embd[0]
            t_encode = time.time() - t0
            
            from_enc_q.put(StreamingMessage(MsgType.MSG_EMBD, data=audio_embd, is_last=msg.is_last, encode_time=t_encode))

# ==========================================
# 3. 辅助函数: Pydub 音频加载
# ==========================================
def load_audio(audio_path, sample_rate=16000, start_second=None, duration=None):
    """加载音频文件并转换为 16kHz PCM，支持按需加载指定片段"""
    from pydub import AudioSegment
    
    # 使用 pydub 的 start_second 和 duration 参数来减少解码量（如果环境支持）
    # 如果环境中的 pydub 不支持这些参数，它们会被忽略或报错，这里通过 kwargs 传递更稳健
    load_kwargs = {
        "frame_rate": sample_rate, 
        "channels": 1
    }
    if start_second is not None: load_kwargs['start_second'] = start_second
    if duration: load_kwargs['duration'] = duration

    audio_segment = AudioSegment.from_file(audio_path, **load_kwargs)

    bit_depth = audio_segment.sample_width * 8
    max_val = float(1 << (bit_depth - 1))
    
    audio = np.array(
        audio_segment
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .get_array_of_samples(),
    ) / max_val

    return audio

# ==========================================
# 4. 核心流式器 (Engine)
# ==========================================
class ChunkSegment:
    def __init__(self, audio_embd):
        self.audio_embd = audio_embd
        self.committed_text = "" # 该片段锁定的稳定文本

class QwenASREngine:
    def __init__(
        self,
        encoder_frontend_path: str,
        encoder_backend_path: str,
        llm_gguf_path: str,
        mel_filters_path: str,
        verbose: bool = True
    ):
        t_start = time.time()
        print(f"--- 初始化 QwenASR 引擎 ---")
        self.verbose = verbose
        
        # 延迟导入 LLM 组件
        from qwen_asr_gguf import llama
        self.llama_mod = llama # keep reference
        
        # 加载 LLM
        if verbose: print(f"正在加载 LLM: {llm_gguf_path}")
        self.model = llama.LlamaModel(llm_gguf_path)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf_path)
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, n_batch=4096, embeddings=False)
        
        # 建立消息队列
        self.to_enc_q = mp.Queue()
        self.from_enc_q = mp.Queue()
        
        # 启动编码器进程
        if verbose: print("正在启动音频编码进程...")
        self.enc_proc = mp.Process(
            target=encoder_worker_proc, 
            args=(self.to_enc_q, self.from_enc_q, encoder_frontend_path, encoder_backend_path, mel_filters_path), 
            daemon=True
        )
        self.enc_proc.start()
        
        # 等待就绪
        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_READY:
            if verbose: print("音频编码进程就绪。")
        
        self.load_time = time.time() - t_start
        
        # 基础 Token ID缓存
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def shutdown(self):
        self.to_enc_q.put(StreamingMessage(MsgType.CMD_STOP))
        msg = self.from_enc_q.get()
        if msg.msg_type == MsgType.MSG_DONE:
            if self.verbose: print("\n编码进程已安全终止。")
        self.enc_proc.join()

    def _run_llm_buffered(self, audio_embd, prefix_text, context, sampler, 
                         rollback_num, is_last_chunk=False, language=None):
        """内部方法：执行具体的 LLM 生成逻辑"""
        import numpy as np
        
        # 1. Prompt Construction
        system_text = "You are a helpful assistant. "
        user_prompt_text = f"{context}\n\n" if context else ""

        def tk(t): return self.model.tokenize(t)

        prefix_tokens = [self.ID_IM_START] + tk(f"system\n{system_text}") + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(f"user\n{user_prompt_text}") + [self.ID_AUDIO_START]
        
        # 构建 Assistant 引导部分
        assistant_prompt = "assistant\n"
        if language:
            assistant_prompt += f"language {language}"

        suffix_tokens = [self.ID_AUDIO_END] + tk("数字用0123456789，语音转录：") + [self.ID_IM_END] + \
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
        batch = self.llama_mod.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start
        
        # 2. Generation Loop
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        cur_pos = total_len
        gen_batch = self.llama_mod.LlamaBatch(4, 0, 1)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        
        last_sampled_token = sampler.sample(self.ctx.ptr)
        
        for _ in range(150): # Max new tokens per chunk
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            
            gen_batch.set_token(last_sampled_token, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            self.ctx.decode(gen_batch)
            
            # --- OVERLAP GAP: CPU Word ---
            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    # 实时打印稳定文本 (如果不需要实时打印，可以注释掉)
                    print(piece, end="", flush=True)
                    stable_text_acc += piece
            # ------------------------------
            
            cur_pos += 1
            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1
            
        gen_time = time.time() - t_gen_start
            
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
            
        return prefix_text + stable_text_acc, stable_tokens, prefill_time, gen_time, total_len, n_gen_tokens

    def transcribe(
        self, 
        audio_file: str, 
        language: str = None, 
        context: str = None, 
        chunk_size: float = 40.0,
        start_second: float = 0.0,
        duration: float = None,
        temperature: float = 0.4,
        memory_num: int = 2,    # 记忆中保留的音频片段数量
        rollback_num: int = 5   # 回滚/撤销的 Token 数量
    ) -> str:
        
        if self.verbose:
            print(f"\n正在处理: {audio_file}")
            print(f"参数配置: 切片={chunk_size}s, 记忆数={memory_num}, 温度={temperature}, 语言={language}, 起始={start_second}s, 时长={duration}s")

        # 重新创建 Sampler (以支持动态 Temperature)
        sampler = self.llama_mod.LlamaSampler(temperature=temperature)

        # 加载音频 (使用 Pydub)
        full_audio = load_audio(audio_file, sample_rate=16000, start_second=start_second, duration=duration)
        sr = 16000

        SAMPLES_PER_CHUNK = int(chunk_size * sr)
        total_len = len(full_audio)
        num_chunks = int(np.ceil(total_len / SAMPLES_PER_CHUNK))
        
        # 状态重置
        segment_queue = deque(maxlen=memory_num)
        archive_text = ""
        total_full_text = ""
        
        # 统计
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "wait_time": 0.0, "encode_time": 0.0
        }
        
        t_main_start = time.time()
        
        # --- 内部 Chunk 获取函数 ---
        def get_chunk(idx):
            s = idx * SAMPLES_PER_CHUNK
            e = min((idx+1) * SAMPLES_PER_CHUNK, total_len)
            chunk = full_audio[s:e]
            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))
            return chunk, (idx == num_chunks - 1)

        print("--- 开始流式转录 ---")
        
        # 1. 发送第一个块
        if num_chunks > 0:
            chunk, is_last = get_chunk(0)
            self.to_enc_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=chunk, is_last=is_last))
        
        for i in range(num_chunks):
            # 2. 等待当前块的 Embedding
            t_w_start = time.time()
            msg: StreamingMessage = self.from_enc_q.get()
            stats["wait_time"] += (time.time() - t_w_start)
            stats["encode_time"] += msg.encode_time
            
            current_embd = msg.data
            was_last = msg.is_last
            
            # 3. 握手触发：立刻发送下一块的编码指令（如果有）
            if not was_last:
                next_chunk, next_is_last = get_chunk(i + 1)
                self.to_enc_q.put(StreamingMessage(MsgType.CMD_ENCODE, data=next_chunk, is_last=next_is_last))
            
            # 4. LLM 解码
            new_seg = ChunkSegment(current_embd)
            if len(segment_queue) >= memory_num:
                oldest = segment_queue.popleft()
                archive_text += oldest.committed_text
            segment_queue.append(new_seg)
            
            prefix_str = archive_text + "".join([s.committed_text for s in list(segment_queue)[:-1]])
            total_audio_input = np.concatenate([s.audio_embd for s in segment_queue], axis=0)
            
            full_out_text, _, t_pre, t_gen, n_pre, n_gen = self._run_llm_buffered(
                total_audio_input, prefix_str, context, sampler, rollback_num, 
                is_last_chunk=was_last, language=language
            )
            
            # 更新 Segment 产生的文本 (减去 Prefix)
            new_text_part = full_out_text[len(prefix_str):]
            new_seg.committed_text = new_text_part
            total_full_text = full_out_text
            
            stats["prefill_time"] += t_pre
            stats["decode_time"] += t_gen
            stats["prefill_tokens"] += n_pre
            stats["decode_tokens"] += n_gen

        t_total = time.time() - t_main_start
        audio_duration = total_len / 16000
        
        print('\n\n')
        print('='*10 + 'ITN处理结果' + '='*10)
        total_full_text = chinese_itn.chinese_to_num(total_full_text)
        print(total_full_text)
        print('='*30)
        
        if self.verbose:
            rtf = t_total / audio_duration if audio_duration > 0 else 0
            prefill_speed = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0
            decode_speed = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0
            
            print(f"\n\n📊 性能统计:")
            print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
            print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
            print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
            print(f"  🔹 编码等待    : {stats['wait_time']:.2f} 秒 (等待音频特征提取)")
            print(f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {prefill_speed:.1f} tokens/s)")
            print(f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {decode_speed:.1f} tokens/s)")
            
        return total_full_text

# ==========================================
# 5. 主程序 (Example Usage)
# ==========================================
if __name__ == "__main__":
    # Windows 环境多进程启动优化
    import warnings
    warnings.filterwarnings("ignore")
    
    # 定义路径
    PROJECT_ROOT = Path(__file__).parent.absolute()
    frontend = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_frontend.fp16.onnx")
    backend = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_backend.fp16.onnx")
    gguf = os.path.join(PROJECT_ROOT, "model", "qwen3_asr_llm.q8_0.gguf")
    mel_filters = os.path.join(PROJECT_ROOT, "model", "mel_filters.npy")

    # 1. 初始化引擎 (包含加载模型，比较耗时)
    print("正在初始化引擎...")
    engine = QwenASREngine(
        encoder_frontend_path=frontend,
        encoder_backend_path=backend,
        llm_gguf_path=gguf,
        mel_filters_path=mel_filters,
        verbose=True
    )

    # 2. 执行转录 (可调用多次)
    audio_path = "睡前消息.m4a"
    
    # 示例：仅转录前 60 秒，分块 40 秒
    result_text = engine.transcribe(
        audio_file=audio_path,
        context="这是1004期睡前消息，主持人叫督工，助理叫静静，",
        language="Chinese", # 强制指定语言 (如 'Chinese', 'English', None)
        start_second=0.0,   # 从何处开始读音频
        duration=120,       # 读取多长音频，None 表示全部读取
        temperature=0.4,    # LLM Decode 温度
        chunk_size=40.0,    # 每一片段的时长
        memory_num=2,       # 记忆多少片段
        rollback_num=5      # 连接处回滚几个 TOKEN
    )
    
    
    # 3. 资源清理
    engine.shutdown()
