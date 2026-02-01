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
ENCODER_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_discrete_all.onnx")
LLM_GGUF_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_asr_llm.f16.gguf")

# Special Token IDs
ID_IM_START = 151644
ID_IM_END = 151645
ID_AUDIO_START = 151669
ID_AUDIO_END = 151670
ID_AUDIO_PAD = 151676
ID_ASR_TEXT = 151704

class Qwen3ASRTranscriber:
    def __init__(self):
        print("--- 初始化音频转录引擎 ---")
        
        # 1. 加载 Processor (用于特征提取)
        print(f"加载 Processor: {HF_MODEL_DIR}")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
        tokenizer = Qwen2TokenizerFast.from_pretrained(HF_MODEL_DIR)
        self.processor = Qwen3ASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        # 2. 加载 Encoder ONNX
        print(f"加载 Encoder ONNX: {ENCODER_ONNX_PATH}")
        self.encoder_sess = ort.InferenceSession(ENCODER_ONNX_PATH, providers=['CPUExecutionProvider'])
        
        # 3. 加载 LLM GGUF
        print(f"加载 GGUF LLM: {LLM_GGUF_PATH}")
        self.model = llama.LlamaModel(LLM_GGUF_PATH, n_gpu_layers=0)
        self.embedding_table = llama.get_token_embeddings_gguf(LLM_GGUF_PATH)
        
        # 4. 创建 Context 和 Sampler
        # n_ctx 设置为足够长 (Prompt 约 200 + Audio 约 1000 + 剩余生成空间)
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, embeddings=False)
        self.sampler = llama.LlamaSampler(temperature=0.4)

    def transcribe(self, audio_path: str, context: Optional[str] = None):
        if not os.path.exists(audio_path):
            print(f"❌ 找不到音频文件: {audio_path}")
            return
            
        print(f"\n开始转录: {audio_path}")
        if context:
            print(f"应用上下文信息: {context}")
        
        # 1. 提取 Mel 特征
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(text="语音转录：", audio=audio, return_tensors="pt")
        mel = inputs.input_features.numpy() # [1, 128, T]
        
        # Transpose to [1, T, 128] for ONNX
        if mel.shape[1] == 128:
            mel = mel.transpose(0, 2, 1)
            
        # 2. 音频编码 (Encoder)
        audio_embd = self.encoder_sess.run(None, {"mel": mel})[0][0] # [T_token, Dim]
        n_audio_tokens = audio_embd.shape[0]
        
        # 3. 准备 LLM Input Embeddings
        # 构造 Prompt Token 序列
        user_prompt = "user\n"
        user_prompt += f"输出数字要小写\n\n"
        
        prefix_tokens = [ID_IM_START] + self.model.tokenize("system\nYou are a helpful assistant.") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize(user_prompt) + [ID_AUDIO_START]
        
        suffix_tokens = [ID_AUDIO_END] + self.model.tokenize("语音转录：") + [ID_IM_END] + \
                        [ID_IM_START] + self.model.tokenize("assistant\n") + [ID_ASR_TEXT]
        
        n_prefix = len(prefix_tokens)
        n_suffix = len(suffix_tokens)
        total_len = n_prefix + n_audio_tokens + n_suffix
        
        # 拼接 Embedding
        full_embd = np.zeros((total_len, self.model.n_embd), dtype=np.float32)
        full_embd[:n_prefix] = self.embedding_table[prefix_tokens]
        full_embd[n_prefix : n_prefix + n_audio_tokens] = audio_embd
        full_embd[n_prefix + n_audio_tokens :] = self.embedding_table[suffix_tokens]
        
        # 4. Prefill (推理)
        # 构造多维位置编码 (M-RoPE)
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        
        # 初始化 Batch 并解码
        batch = llama.LlamaBatch(max(total_len, 2048), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        self.ctx.clear_kv_cache() # 每次新转录重置 Context
        if self.ctx.decode(batch) != 0:
            print("❌ Prefill 失败")
            return
            
        # 5. 递归生成文字
        gen_batch = llama.LlamaBatch(4, 0, 1) # 4 个位置平面
        cur_pos = total_len
        result_text = ""
        
        print("Result: ", end="", flush=True)
        for i in range(512): # 最大 512 字
            token_id = self.sampler.sample(self.ctx.ptr)
            if token_id == self.model.eos_token or token_id == ID_IM_END:
                break
                
            piece = self.model.token_to_bytes(token_id).decode('utf-8', errors='replace')
            print(piece, end="", flush=True)
            result_text += piece
            
            # 喂回 Token 进行下一步预测
            gen_batch.set_token(token_id, pos=np.array([cur_pos, cur_pos, cur_pos, 0], dtype=np.int32))
            if self.ctx.decode(gen_batch) != 0:
                break
            cur_pos += 1
            
        print("\n--- 转录结束 ---")
        return result_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-ASR (ONNX + GGUF) 离线转录工具")
    parser.add_argument("input", help="音频文件路径 (如 test.mp3)")
    parser.add_argument("--context", help="转录上下文信息 (有助于提高准确度)", default=None)
    args = parser.parse_args()
    
    # 临时抑制冗余日志
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    engine = Qwen3ASRTranscriber()
    engine.transcribe(args.input, context=args.context)
