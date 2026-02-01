import os
import sys
import torch
import numpy as np
import onnxruntime as ort
import librosa
from pathlib import Path
from typing import Optional
from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast

# 添加本地 Qwen-ASR GGUF 适配库路径
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from qwen_asr_gguf import llama
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

# ==========================================
# 配置参数 (根据您的路径环境修改)
# ==========================================
HF_MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
FRONTEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_frontend.onnx")
BACKEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_backend.int8.onnx")
LLM_GGUF_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_asr_llm.q8_0.gguf")

# Special Token IDs
ID_IM_START = 151644
ID_IM_END = 151645
ID_AUDIO_START = 151669
ID_AUDIO_END = 151670
ID_AUDIO_PAD = 151676
ID_ASR_TEXT = 151704

class Qwen3ASRTranscriber:
    def __init__(self):
        print("--- 初始化音频转录引擎 (模块化 Encoder) ---")
        
        # 1. 加载 Processor (用于特征提取)
        print(f"加载 Processor: {HF_MODEL_DIR}")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
        tokenizer = Qwen2TokenizerFast.from_pretrained(HF_MODEL_DIR)
        self.processor = Qwen3ASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        # 2. 加载模块化 Encoder ONNX
        providers = ['CPUExecutionProvider']
        print(f"加载 Encoder Frontend: {FRONTEND_ONNX_PATH}")
        self.frontend_sess = ort.InferenceSession(FRONTEND_ONNX_PATH, providers=providers)
        print(f"加载 Encoder Backend: {BACKEND_ONNX_PATH}")
        self.backend_sess = ort.InferenceSession(BACKEND_ONNX_PATH, providers=providers)
        
        # 3. 加载 LLM GGUF
        print(f"加载 GGUF LLM: {LLM_GGUF_PATH}")
        self.model = llama.LlamaModel(LLM_GGUF_PATH)
        self.embedding_table = llama.get_token_embeddings_gguf(LLM_GGUF_PATH)
        
        # 4. 创建 Context 和 Sampler
        # n_ctx: 上下文长度 (8192 tokens 约可支持 10 分钟音频 + 对话)
        # n_batch: 一次推入的最大 Token 数 (必须 >= n_ctx 以支持一次性 Prefill)
        MAX_CTX = 8192
        self.ctx = llama.LlamaContext(self.model, n_ctx=MAX_CTX, n_batch=MAX_CTX, embeddings=False)
        self.sampler = llama.LlamaSampler(temperature=0.4) 

    def transcribe(self, audio_path: str, context: Optional[str] = None):
        if not os.path.exists(audio_path):
            print(f"❌ 找不到音频文件: {audio_path}")
            return
            
        print(f"\n开始转录: {audio_path}")
        
        # 1. 提取 Mel 特征
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(text="语音转录：", audio=audio, return_tensors="pt")
        mel = inputs.input_features.numpy()
        if mel.shape[1] == 128:
            mel = mel.transpose(0, 2, 1)
            
        # 2. 模块化音频编码 (Modular Encoding)
        # Step A: Frontend (卷积)
        feat_out = self.frontend_sess.run(None, {"mel": mel})[0]
        
        # Step B: Backend (Transformer)
        audio_embd = self.backend_sess.run(None, {"feat_in": feat_out})[0]
        if audio_embd.ndim == 3:
            audio_embd = audio_embd[0] # [1, T, D] -> [T, D]
        
        # 新的 Backend 已经去除了 Overlap，无需手动切片
        n_audio_tokens = audio_embd.shape[0]
        
        # 3. 准备 LLM Input Embeddings
        # 如果提供了 Context，将其拼接到 User Prompt 中
        user_prompt_text = ''
        if context:
            user_prompt_text = f"这是当前对话的上下文信息：{context}\n\n"

        prefix_tokens = [ID_IM_START] + self.model.tokenize("system\nYou are a helpful assistant.") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize(f"user\n{user_prompt_text}") + [ID_AUDIO_START]
        
        suffix_tokens = [ID_AUDIO_END] + self.model.tokenize("语音转录：") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize("assistant\n") + [ID_ASR_TEXT]
        
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
        
        batch = llama.LlamaBatch(max(total_len, 2048), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        self.ctx.clear_kv_cache()
        if self.ctx.decode(batch) != 0:
            print("❌ Prefill 失败")
            return
            
        # 5. 递归生成文字
        gen_batch = llama.LlamaBatch(4, 0, 1)
        cur_pos = total_len
        result_text = ""
        
        print("Result: ", end="", flush=True)
        for i in range(512):
            token_id = self.sampler.sample(self.ctx.ptr)
            if token_id == self.model.eos_token or token_id == ID_IM_END:
                break
                
            piece = self.model.token_to_bytes(token_id).decode('utf-8', errors='replace')
            print(piece, end="", flush=True)
            result_text += piece
            
            gen_batch.set_token(token_id, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            if self.ctx.decode(gen_batch) != 0:
                break
            cur_pos += 1
            
        print("\n--- 转录结束 ---")
        return result_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-ASR (Modular ONNX + GGUF) 离线转录工具")
    parser.add_argument("input", help="音频文件路径 (如 test.mp3)")
    parser.add_argument("--context", help="转录上下文信息 (有助于提高准确度)", default=None)
    args = parser.parse_args()
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    engine = Qwen3ASRTranscriber()
    engine.transcribe(args.input, context=args.context)
