import time
import os
import sys
import codecs
# import torch
import numpy as np
import onnxruntime as ort
import librosa
from pathlib import Path
from typing import Optional, List
# from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast <-- Removed

# 添加本地 Qwen-ASR-GGUF 适配库路径
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from qwen_asr_gguf import llama

# ==================== Vulkan 选项 ====================

# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "1"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）

providers = ['DmlExecutionProvider']
# providers = ['CPUExecutionProvider']


# ==========================================
# 配置参数 (根据您的路径环境修改)
# ==========================================
FRONTEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_frontend.fp32.onnx")
BACKEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_backend.fp32.onnx")
LLM_GGUF_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_asr_llm.q8_0.gguf")

# Special Token IDs
ID_IM_START = 151644
ID_IM_END = 151645
ID_AUDIO_START = 151669
ID_AUDIO_END = 151670
ID_AUDIO_PAD = 151676
ID_ASR_TEXT = 151704

class FastWhisperMel:
    """完全基于 NumPy 和 Librosa 的 Mel 提取器 (替代 Transformers)"""
    def __init__(self, filter_path):
        self.filters = np.load(filter_path) # (201, 128)
        
    def __call__(self, audio):
        # 1. STFT (Reflect padding, Hann window)
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
        # 2. Power Spectrum
        magnitudes = np.abs(stft) ** 2 
        # 3. Mel Filterbank
        mel_spec = np.dot(self.filters.T, magnitudes) # (128, 201) x (201, T) -> (128, T)
        # 4. Log10
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        # 5. Dynamic Range Compression
        log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
        # 6. Scaling
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.T.astype(np.float32) # (T, 128)

class Qwen3ASRTranscriber:
    def __init__(self):
        t0 = time.time()
        print("--- 初始化音频转录引擎 (模块化 Encoder) ---")
        
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        # 1. 加载 Mel 滤波器
        print(f"加载 Custom Mel Filters...")
        # 假设 filter 在 model/mel_filters.npy，如果不存在需要提示生成
        filter_path = os.path.join(PROJECT_ROOT, "model", "mel_filters.npy")
        if not os.path.exists(filter_path):
             raise FileNotFoundError(f"找不到 Mel 滤波器文件: {filter_path}。请先运行 export_mel_filters.py")
        self.mel_extractor = FastWhisperMel(filter_path)
        
        
        # 2. 加载模块化 Encoder ONNX
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"加载 Encoder Frontend: {FRONTEND_ONNX_PATH}")
        self.frontend_sess = ort.InferenceSession(FRONTEND_ONNX_PATH, sess_options=sess_options, providers=providers)
        print(f"加载 Encoder Backend: {BACKEND_ONNX_PATH}")
        self.backend_sess = ort.InferenceSession(BACKEND_ONNX_PATH, sess_options=sess_options, providers=providers)
        
        # 模型预热 (Warmup)
        dummy_mel = np.random.randn(1, 200, 128).astype(np.float32)
        dummy_feat = self.frontend_sess.run(None, {"mel": dummy_mel})[0]
        self.backend_sess.run(None, {"feat_in": dummy_feat})
        
        # 3. 加载 LLM GGUF
        print(f"加载 GGUF LLM: {LLM_GGUF_PATH}")
        self.model = llama.LlamaModel(LLM_GGUF_PATH)
        self.embedding_table = llama.get_token_embeddings_gguf(LLM_GGUF_PATH)
        
        # 更新 Token IDs 为原生查询结果
        global ID_IM_START, ID_IM_END, ID_AUDIO_START, ID_AUDIO_END, ID_ASR_TEXT
        ID_IM_START = self.model.token_to_id("<|im_start|>")
        ID_IM_END = self.model.token_to_id("<|im_end|>")
        ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

        # 4. 创建 Context 和 Sampler
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, n_batch=4096, embeddings=False)
        self.sampler = llama.LlamaSampler(temperature=0) 
        
        self.t_load_duration = time.time() - t0

    def transcribe(self, audio_path: str, context: str = "", language: str = None):
        if not os.path.exists(audio_path):
            print(f"❌ 找不到音频文件: {audio_path}")
            return
        print(f"--- Processing {audio_path} ---")
        
        # 1. 提取 Mel 特征
        audio, _ = librosa.load(audio_path, sr=16000)
        mel = self.mel_extractor(audio) # [T, 128]
        mel = mel[np.newaxis, ...] # [1, T, 128]
            
        t_encoder_start = time.time()
        
        # 2. 编码器前馈
        # Step A: Frontend (卷积)
        t_front_start = time.time()
        feat_out = self.frontend_sess.run(None, {"mel": mel})[0]
        t_front_end = time.time()
        
        # Step B: Backend (Transformer)
        t_back_start = time.time()
        audio_embd = self.backend_sess.run(None, {"feat_in": feat_out})[0]
        if audio_embd.ndim == 3: audio_embd = audio_embd[0] # [T_ds, 1024]
        t_back_end = time.time()
        
        t_encoder_end = time.time()
        n_audio_tokens = audio_embd.shape[0]
        
        # 3. 准备 LLM Input Embeddings
        # Prompt 格式:
        # <|im_start|>system\n...<|im_end|>
        # <|im_start|>user\n{context}\n\n<|audio_start|>{Audio Embeds}<|audio_end|>语音转录：<|im_end|>
        # <|im_start|>assistant\n
        
        user_prompt_text = ''
        if context:
            user_prompt_text = f"{context}\n\n"

        prefix_tokens = [ID_IM_START] + self.model.tokenize("system\nYou are a helpful assistant.") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize(f"user\n{user_prompt_text}") + [ID_AUDIO_START]
        
        # 构建 Assistant 引导部分
        # 格式: <|audio_end|>语音转录：<|im_end|><|im_start|>assistant\n[language {Lang}]<asr_text>
        assistant_prompt = "assistant\n"
        if language:
            assistant_prompt += f"language {language}"
            
        suffix_tokens = [ID_AUDIO_END] + self.model.tokenize("语音转录：") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize(assistant_prompt) + [ID_ASR_TEXT]
        
        n_prefix = len(prefix_tokens)
        n_suffix = len(suffix_tokens)
        total_len = n_prefix + n_audio_tokens + n_suffix
        
        full_embd = np.zeros((total_len, self.model.n_embd), dtype=np.float32)
        full_embd[:n_prefix] = self.embedding_table[prefix_tokens]
        full_embd[n_prefix : n_prefix + n_audio_tokens] = audio_embd
        full_embd[n_prefix + n_audio_tokens :] = self.embedding_table[suffix_tokens]
        
        # 4. Prefill (推理)
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        
        # Qwen3 使用 M-RoPE 位置编码，每个 token 需要 4 个位置值。
        # 因此，我们需要分配 4 倍的 Batch Pos 空间以防止 Heap Corruption (Buffer Overflow)。
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        print(f"DEBUG: Starting Prefill. total_len={total_len}, batch_size={max(total_len, 2048)}")
        self.ctx.clear_kv_cache()
        
        t_prefill_start = time.time()
        try:
            if self.ctx.decode(batch) != 0:
                print("❌ Prefill 失败 (decode returned non-zero)")
                return
        except Exception as e:
            print(f"❌ Prefill 异常: {e}")
            raise e
        t_prefill_end = time.time()
        
        print("DEBUG: Prefill successful. Starting Generation.")
            
        # 5. 递归生成文字
        gen_batch = llama.LlamaBatch(4, 0, 1)
        cur_pos = total_len
        result_text = ""
        
        t_decode_start = time.time()
        n_decode_tokens = 0
        
        print("Result: ", end="", flush=True)
        
        # BPE 增量解码器 (处理多字节字符分割)
        decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
        
        for i in range(512):
            token_id = self.sampler.sample(self.ctx.ptr)
            if token_id == self.model.eos_token or token_id == ID_IM_END:
                break
            
            # 优先提交下一次解码任务 (Compute First)
            gen_batch.set_token(token_id, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            ret = self.ctx.decode(gen_batch)
            cur_pos += 1
            n_decode_tokens += 1
            
            # 后处理耗时操作 (IO / Text Processing)
            # 获取 Token 字节并流式解码
            delta = decoder.decode(self.model.token_to_bytes(token_id), final=False)
            if delta:
                print(delta, end="", flush=True)
                result_text += delta
            
            # 检查解码结果
            if ret != 0:
                print(f"❌ Decode Error: {ret}")
                break
        
        
        t_decode_end = time.time()
            
        print("\n--- 转录结束 ---")
        
        # 统计报告
        t_encoder_total = t_encoder_end - t_encoder_start
        t_prefill_total = t_prefill_end - t_prefill_start
        t_decode_total = t_decode_end - t_decode_start
        
        # 计算总耗时 (从 Encoder 开始到 Decode 结束)
        t_total_transcribe = t_encoder_total + t_prefill_total + t_decode_total
        
        # 获取音频时长 (seconds)
        audio_duration = len(audio) / 16000
        rtf = t_total_transcribe / audio_duration if audio_duration > 0 else 0
        
        prefill_cps = total_len / t_prefill_total if t_prefill_total > 0 else 0
        decode_cps = n_decode_tokens / t_decode_total if t_decode_total > 0 else 0
        
        print(f"\n📊 性能统计:")
        print(f"  🔹 加载耗时: {self.t_load_duration:.3f} s")
        print(f"  🔹 音频时长: {audio_duration:.2f} s")
        print(f"  🔹 总耗时  : {t_total_transcribe:.3f} s (RTF: {rtf:.4f})")
        print(f"  🔹 - 音频编码: {t_encoder_total:.3f} s (Frontend: {t_front_end - t_front_start:.3f}s, Backend: {t_back_end - t_back_start:.3f}s)")
        print(f"  🔹 - Prefill : {t_prefill_total:.3f} s | {total_len} tokens | {prefill_cps:.1f} tokens/s")
        print(f"  🔹 - Decode  : {t_decode_total:.3f} s | {n_decode_tokens} tokens | {decode_cps:.1f} tokens/s")
        
        return result_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-ASR (Modular ONNX + GGUF) 离线转录工具")
    parser.add_argument("input", help="音频文件路径 (如 test.mp3)")
    parser.add_argument("--context", help="转录上下文信息 (有助于提高准确度)", default=None)
    parser.add_argument("--language", help="强制指定语言 (如 'Chinese', 'English')", default=None)
    args = parser.parse_args()
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    engine = Qwen3ASRTranscriber()
    engine.transcribe(args.input, context=args.context, language=args.language)
