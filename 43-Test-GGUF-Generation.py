import os
import torch
import numpy as np
from qwen_asr_gguf import llama

# 配置
GGUF_PATH = "model/qwen3_asr_llm.f16.gguf"
EMBEDS_PATH = "capture_llm/llm_input_embeds.npy"
MAX_NEW_TOKENS = 20

def main():
    print(f"--- 正在进行 GGUF LLM 递归解码测试 ---")
    
    # 1. 加载捕获的 Embedding
    if not os.path.exists(EMBEDS_PATH):
        print(f"❌ 找不到捕捉到的数据: {EMBEDS_PATH}")
        return
    embeds = np.load(EMBEDS_PATH)
    n_tokens = embeds.shape[1]
    hidden_size = embeds.shape[2]
    print(f"Input Embeds Shape: {embeds.shape}")

    # 2. 加载模型
    print(f"Loading GGUF model: {GGUF_PATH}")
    model = llama.LlamaModel(GGUF_PATH, n_gpu_layers=0) # 使用 CPU
    # 扩大 n_ctx 以容纳 prompt + new tokens
    ctx = llama.LlamaContext(model, n_ctx=n_tokens + MAX_NEW_TOKENS + 10, embeddings=True)
    
    # 3. 创建 Batch 并执行 Prefill
    # Prefill 使用 Embedding
    prefill_batch = llama.LlamaBatch(max(n_tokens, 2048), embeds.shape[2], 1)
    
    # 构造 M-RoPE 专用的位置编码 (3层线性 Pos + 1层 Zero)
    pos_base = np.arange(0, n_tokens, dtype=np.int32)
    pos_arr = np.concatenate([
        pos_base,           # Temporal/Sequence
        pos_base,           # Height
        pos_base,           # Width
        np.zeros(n_tokens, dtype=np.int32) # Reserved/Zero
    ])
    
    prefill_batch.set_embd(embeds[0], pos=pos_arr)
    
    print("Prefilling with prompt embeddings...")
    ret = ctx.decode(prefill_batch)
    if ret != 0:
        print(f"❌ Prefill Decode 失败: {ret}")
        return

    # 4. 递归生成
    sampler = llama.LlamaSampler(temperature=0) # Greedy
    
    # 创建专门用于生成步的小 Batch (Token 模式，embd_dim=0)
    # 注意：为了能存 4 层 pos，n_tokens 设为 4
    gen_batch = llama.LlamaBatch(4, 0, 1)
    
    print("\n--- 生成结果 ---")
    generated_text = ""
    cur_pos = n_tokens
    
    for i in range(MAX_NEW_TOKENS):
        # 采样下一个 Token
        token_id = sampler.sample(ctx.ptr)
        
        # 转换为文字
        piece = model.token_to_bytes(token_id).decode('utf-8', errors='replace')
        print(piece, end="", flush=True)
        generated_text += piece
        
        if token_id == model.eos_token:
            print("\n[EOS 停止]")
            break
            
        # 解码这一个 Token
        gen_batch.n_tokens = 1
        gen_batch.struct.token[0] = token_id
        # 填充 4 维位置
        for p_idx in range(3):
            gen_batch.pos[p_idx] = cur_pos
        gen_batch.pos[3] = 0
        gen_batch.n_seq_id[0] = 1
        gen_batch.seq_id[0][0] = 0 # seq_id
        gen_batch.logits[0] = 1 # 我们需要最后一个（也是唯一的）token 的 logits
        
        ret = ctx.decode(gen_batch)
        if ret != 0:
            print(f"\n❌ Step {i} Decode 失败: {ret}")
            break
            
        cur_pos += 1
        
    print("\n\n测试完成。")

if __name__ == "__main__":
    main()
