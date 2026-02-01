import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import ctypes

# 添加路径
PROJECT_ROOT = Path(__file__).parent.absolute()
from qwen_asr_gguf import llama

# 配置
ENCODER_PATH = "model/onnx/qwen3_asr_encoder_discrete.onnx"
GGUF_PATH = "model/qwen3_asr_llm.f16.gguf"
MEL_PATH = "test_mel.npy"

# Token IDs (From tokenizer_config.json)
ID_IM_START = 151644
ID_IM_END = 151645
ID_AUDIO_START = 151669
ID_AUDIO_END = 151670
ID_AUDIO_PAD = 151676
ID_ASR_TEXT = 151704

def main():
    print("--- 正在运行 ASR 联合推理测试 (Encoder ONNX + LLM GGUF) ---")
    
    # 1. 加载 Mel
    if not os.path.exists(MEL_PATH):
        print(f"❌ 找不到 Mel 文件: {MEL_PATH}")
        return
    mel = np.load(MEL_PATH)
    if mel.shape[1] == 128 and mel.shape[2] != 128:
        mel = mel.transpose(0, 2, 1) # [1, 128, 1000] -> [1, 1000, 128]
    print(f"Loaded Mel Shape: {mel.shape}")

    # 2. 运行 Encoder ONNX
    print(f"Loading Encoder: {ENCODER_PATH}")
    encoder_sess = ort.InferenceSession(ENCODER_PATH, providers=['CPUExecutionProvider'])
    audio_embd = encoder_sess.run(None, {"mel": mel})[0][0] # [T_audio, Dim]
    n_audio_tokens = audio_embd.shape[0]
    print(f"Encoder Output Shape: {audio_embd.shape} (Tokens: {n_audio_tokens})")

    # 3. 加载 GGUF
    print(f"Loading GGUF LLM: {GGUF_PATH}")
    model = llama.LlamaModel(GGUF_PATH, n_gpu_layers=0)
    embedding_table = llama.get_token_embeddings_gguf(GGUF_PATH)
    
    # 4. 构建 Prompt
    # 结构: system + user(audio tokens) + assistant(asr start)
    prefix_tokens = [ID_IM_START] + model.tokenize("system\nYou are a helpful assistant.") + [ID_IM_END] + \
                    [ID_IM_START] + model.tokenize("user\n") + [ID_AUDIO_START]
    
    mid_tokens = [ID_AUDIO_PAD] * n_audio_tokens
    
    suffix_tokens = [ID_AUDIO_END] + model.tokenize("语音转录：") + [ID_IM_END] + \
                    [ID_IM_START] + model.tokenize("assistant\n") + [ID_ASR_TEXT]
    
    n_prefix = len(prefix_tokens)
    n_suffix = len(suffix_tokens)
    total_input_len = n_prefix + n_audio_tokens + n_suffix
    print(f"Prompt Config: Prefix={n_prefix}, Audio={n_audio_tokens}, Suffix={n_suffix}, Total={total_input_len}")

    # 5. 准备 Embeddings
    full_embd = np.zeros((total_input_len, model.n_embd), dtype=np.float32)
    full_embd[:n_prefix] = embedding_table[prefix_tokens]
    full_embd[n_prefix : n_prefix + n_audio_tokens] = audio_embd
    full_embd[n_prefix + n_audio_tokens :] = embedding_table[suffix_tokens]

    # 6. 推理
    ctx = llama.LlamaContext(model, n_ctx=total_input_len + 512, embeddings=True)
    batch = llama.LlamaBatch(max(total_input_len, 2048), model.n_embd, 1)
    
    # 构造位置编码 (Qwen3 M-RoPE)
    pos_base = np.arange(0, total_input_len, dtype=np.int32)
    pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_input_len, dtype=np.int32)])
    
    batch.set_embd(full_embd, pos=pos_arr)
    
    print("Prefilling...")
    if ctx.decode(batch) != 0:
         print("❌ Prefill failed")
         return

    # 7. 递归生成
    sampler = llama.LlamaSampler(temperature=0)
    gen_batch = llama.LlamaBatch(4, 0, 1) # 4 slots for M-RoPE
    
    print("\n--- 转录结果 ---")
    cur_pos = total_input_len
    for i in range(256):
        token_id = sampler.sample(ctx.ptr)
        if token_id == model.eos_token or token_id == ID_IM_END:
            break
            
        piece = model.token_to_bytes(token_id).decode('utf-8', errors='replace')
        print(piece, end="", flush=True)
        
        # Decode Step
        gen_batch.n_tokens = 1
        gen_batch.struct.token[0] = token_id
        for p_idx in range(3):
            gen_batch.pos[p_idx] = cur_pos
        gen_batch.pos[3] = 0
        gen_batch.n_seq_id[0] = 1
        gen_batch.seq_id[0][0] = 0
        gen_batch.logits[0] = 1
        
        if ctx.decode(gen_batch) != 0:
            print("\n❌ Decode step failed")
            break
        cur_pos += 1

    print("\n\n--- 推理完成 ---")

if __name__ == "__main__":
    main()
