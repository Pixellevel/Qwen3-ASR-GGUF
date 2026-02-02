
import os
import sys
import time
import torch
import numpy as np
import onnxruntime as ort
import librosa
from pathlib import Path
from typing import Optional
from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from qwen_asr_gguf import llama
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

# ==========================================
# 配置参数
# ==========================================
HF_MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
FRONTEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_frontend.onnx")
BACKEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_backend.int8.onnx")
LLM_GGUF_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_asr_llm.q8_0.gguf")

# ONNX Providers
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

# Special Token IDs
ID_IM_START = 151644
ID_IM_END = 151645
ID_AUDIO_START = 151669
ID_AUDIO_END = 151670
ID_ASR_TEXT = 151704

class Qwen3ASRIncrementalEncoderStreamer:
    def __init__(self):
        print("--- 初始化 ONNX/GGUF 增量 Encoder 流式引擎 ---")
        
        # 1. Processor
        feature_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
        tokenizer = Qwen2TokenizerFast.from_pretrained(HF_MODEL_DIR, fix_mistral_regex=True)
        self.processor = Qwen3ASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        # 2. Encoder ONNX
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.frontend_sess = ort.InferenceSession(FRONTEND_ONNX_PATH, sess_options=sess_options, providers=providers)
        self.backend_sess = ort.InferenceSession(BACKEND_ONNX_PATH, sess_options=sess_options, providers=providers)
        
        # 3. LLM GGUF
        self.model = llama.LlamaModel(LLM_GGUF_PATH)
        self.embedding_table = llama.get_token_embeddings_gguf(LLM_GGUF_PATH)
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, n_batch=2048, embeddings=False)
        self.sampler = llama.LlamaSampler(temperature=0.0)

    def encode_incremental(self, audio_chunk):
        """
        只编码给定的这段音频，不考虑历史，特征由调用者拼接。
        """
        # Mel 提取 (注意：这里的 processor 处理的是一整段，由于我们已经手动切分了音频，它只处理这一段)
        inputs = self.processor(text="语音转录：", audio=audio_chunk, return_tensors="pt")
        mel = inputs.input_features.numpy()
        if mel.shape[1] == 128:
            mel = mel.transpose(0, 2, 1)
            
        # Encoder
        feat_out = self.frontend_sess.run(None, {"mel": mel})[0]
        audio_embd = self.backend_sess.run(None, {"feat_in": feat_out})[0]
        if audio_embd.ndim == 3:
            audio_embd = audio_embd[0]
        return audio_embd

    def generate_with_features(self, audio_embd_accum, prefix_text=""):
        """
        使用累积的特征进行 LLM 生成。
        """
        n_audio_tokens = audio_embd_accum.shape[0]
        
        # Token 拼接
        prefix_tokens = [ID_IM_START] + self.model.tokenize("system\n语音转录：") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize("user\n") + [ID_AUDIO_START]
        
        suffix_tokens_base = [ID_AUDIO_END] + [ID_IM_END] + [ID_IM_START] + \
                             self.model.tokenize("assistant\n") + [ID_ASR_TEXT]
        
        rollback_tokens = self.model.tokenize(prefix_text) if prefix_text else []
        suffix_tokens = suffix_tokens_base + rollback_tokens
        
        n_prefix = len(prefix_tokens)
        n_suffix = len(suffix_tokens)
        total_len = n_prefix + n_audio_tokens + n_suffix
        
        # 构建 Embedding 会话
        full_embd = np.zeros((total_len, self.model.n_embd), dtype=np.float32)
        full_embd[:n_prefix] = self.embedding_table[prefix_tokens]
        full_embd[n_prefix : n_prefix + n_audio_tokens] = audio_embd_accum
        full_embd[n_prefix + n_audio_tokens :] = self.embedding_table[suffix_tokens]
        
        # Prefill
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        self.ctx.clear_kv_cache()
        if self.ctx.decode(batch) != 0:
            return "", []
            
        # Decode
        gen_batch = llama.LlamaBatch(4, 0, 1)
        cur_pos = total_len
        result_tokens = []
        result_text = ""
        
        for i in range(128):
            token_id = self.sampler.sample(self.ctx.ptr)
            if token_id == self.model.eos_token or token_id == ID_IM_END:
                break
                
            piece = self.model.token_to_bytes(token_id).decode('utf-8', errors='replace')
            print(piece, end="", flush=True)
            result_text += piece
            result_tokens.append(token_id)
            
            gen_batch.set_token(token_id, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            if self.ctx.decode(gen_batch) != 0:
                break
            cur_pos += 1
            
        return result_text, rollback_tokens + result_tokens

def main():
    AUDIO_FILE = "test40.mp3"
    
    # 精确对齐参数
    # 1. Whisper Mel Stride = 160 samples
    # 2. Qwen3 Conv Stride = 2^3 = 8
    # 3. Total Stride = 160 * 8 = 1280 samples/token
    SAMPLES_PER_TOKEN = 1280
    
    CHUNK_TOKENS = 50 # 精确 4.0s (50 * 1280 = 64000 samples)

    ROLLBACK_TOKENS_COUNT = 5
    UNFIXED_CHUNK_NUM = 1
    
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found")
        return

    engine = Qwen3ASRIncrementalEncoderStreamer()
    
    print(f"Loading audio: {AUDIO_FILE}")
    full_audio, _ = librosa.load(AUDIO_FILE, sr=16000)
    
    audio_features_accum = np.zeros((0, engine.model.n_embd), dtype=np.float32)
    
    chunk_samples = CHUNK_TOKENS * SAMPLES_PER_TOKEN

    
    total_chunks = int(np.ceil(len(full_audio) / chunk_samples))
    last_all_tokens = []
    
    print(f"\n--- Starting Incremental Encoder Streaming (Aligned: {CHUNK_TOKENS} tokens/chunk) ---")
    
    for i in range(total_chunks):
        t_start = time.time()
        
        # 1. 增量音频获取 (精确样本对齐)
        start_raw = i * chunk_samples
        end_raw = min((i + 1) * chunk_samples, len(full_audio))
        
        cur_audio_input = full_audio[start_raw:end_raw]

        t_enc_start = time.time()
        # 2. 增量 Encoder 推理
        new_embd = engine.encode_incremental(cur_audio_input)
        
        # 累积特征
        audio_features_accum = np.concatenate([audio_features_accum, new_embd], axis=0)
        t_enc_end = time.time()

        # 3. 计算 Prefix
        prefix_text = ""
        if i >= UNFIXED_CHUNK_NUM and last_all_tokens:
            keep_len = max(0, len(last_all_tokens) - ROLLBACK_TOKENS_COUNT)
            keep_tokens = last_all_tokens[:keep_len]
            prefix_text = ""
            for tid in keep_tokens:
                prefix_text += engine.model.token_to_bytes(tid).decode('utf-8', errors='replace')
            while prefix_text and '\ufffd' in prefix_text:
                 prefix_text = prefix_text[:-1]

        print(f"\n[{i+1}/{total_chunks}] {len(full_audio[0:end_raw])/16000:.1f}s | Enc: {t_enc_end - t_enc_start:.2f}s | Prefix: '{prefix_text}'")
        print("Output: ", end="", flush=True)
        
        # 4. LLM 推理
        new_text, all_tokens = engine.generate_with_features(audio_features_accum, prefix_text)
        last_all_tokens = all_tokens
        
        print(f"\nTotal Step Cost: {time.time() - t_start:.2f}s")

    print("\n--- Streaming Done ---")

if __name__ == "__main__":
    main()
