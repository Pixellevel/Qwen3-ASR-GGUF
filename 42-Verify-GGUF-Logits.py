import os
import sys
import numpy as np
from pathlib import Path

# 添加 fun_asr_gguf 路径以便导入 llama
PROJECT_ROOT = Path(__file__).parent.absolute()

from qwen_asr_gguf import llama

# 配置
GGUF_PATH = "model/qwen3_asr_llm.f16.gguf"
CAPTURE_DIR = Path("capture_llm")

def verify_gguf_logits():
    print("--- 正在进行 GGUF LLM 位准确验证 ---")
    
    embeds_path = CAPTURE_DIR / "llm_input_embeds.npy"
    logits_path = CAPTURE_DIR / "llm_output_logits.npy"
    
    if not embeds_path.exists() or not logits_path.exists():
        print(f"❌ 找不到捕捉的数据: {CAPTURE_DIR}")
        return

    # 1. 加载数据
    # embeds: [1, Seq, 1024/4096]
    # logits: [1, Seq, Vocab]
    embeds = np.load(embeds_path).astype(np.float32)
    ref_logits = np.load(logits_path).astype(np.float32)
    
    print(f"Input Embeds Shape: {embeds.shape}")
    print(f"Reference Logits Shape: {ref_logits.shape}")
    
    # 2. 加载 GGUF 模型
    if not Path(GGUF_PATH).exists():
        print(f"❌ 找不到 GGUF 模型: {GGUF_PATH}")
        return
        
    print(f"Loading GGUF model: {GGUF_PATH}")
    model = llama.LlamaModel(GGUF_PATH, n_gpu_layers=0) # 验证时使用 CPU 保证一致性
    ctx = llama.LlamaContext(model, n_ctx=2048)
    
    # 3. 注入 Embedding
    # GGUF 期望 batch 数据。我们使用 embeds.squeeze(0) 即 [Seq, Dim]
    n_tokens = embeds.shape[1]
    
    # 4. 构造 M-RoPE 专用的位置编码 (3层线性 Pos + 1层 Zero)
    # 对于 Qwen3-VL/ASR 架构，llama.cpp 期望每个 token 对应 4 个位置索引
    pos_base = np.arange(0, n_tokens, dtype=np.int32)
    pos_arr = np.concatenate([
        pos_base,           # Temporal/Sequence
        pos_base,           # Height
        pos_base,           # Width
        np.zeros(n_tokens, dtype=np.int32) # Reserved/Zero
    ])
    print(f"Position Array Shape: {pos_arr.shape}")
    
    # 初始化 Batch。注意：由于我们要注入 4 倍长度的 pos，Batch 容量最好设大点防止溢出
    # 虽然 n_tokens 是 795，但 pos buffer 是由构造函数分配的。
    batch = llama.LlamaBatch(max(n_tokens * 4, 2048), embeds.shape[2], 1)
    batch.set_embd(embeds[0], pos=pos_arr)
    
    # 5. 解码
    print("Running GGUF decode...")
    ret = ctx.decode(batch)
    if ret != 0:
        print(f"❌ Decode 失败: {ret}")
        return
        
    # 6. 获取 Logits 并对比
    # ctx.get_logits() 返回最后一个 token 的 logits [Vocab]
    gguf_logits = ctx.get_logits()
    # 官方 ref_logits 也是 full sequence 的，我们需要取最后一个 token
    ref_last_logits = ref_logits[0, -1, :]
    n_vocab = ref_last_logits.shape[0]
    
    # 将 gguf_logits 转为 numpy
    import ctypes
    gguf_logits_array = np.ctypeslib.as_array(gguf_logits, shape=(n_vocab,)).astype(np.float32)
    
    # 6. 计算误差
    mse = np.mean((gguf_logits_array - ref_last_logits) ** 2)
    mae = np.mean(np.abs(gguf_logits_array - ref_last_logits))
    
    print(f"\n结果对比 (最后一个 Token):")
    print(f"GGUF Logits Min/Max: {gguf_logits_array.min():.4f} / {gguf_logits_array.max():.4f}")
    print(f"Ref  Logits Min/Max: {ref_last_logits.min():.4f} / {ref_last_logits.max():.4f}")
    print(f"Mean Squared Error (MSE): {mse:.2e}")
    print(f"Max Absolute Error (MAE): {mae:.2e}")
    
    if mse < 1e-4:
        print("\n✅ GGUF LLM 与官方模型 Logits 一致性验证通过!")
    else:
        print("\n⚠️ 存在显著差异，请检查 GGUF 转换逻辑或 Position IDs 处理")

if __name__ == "__main__":
    verify_gguf_logits()
