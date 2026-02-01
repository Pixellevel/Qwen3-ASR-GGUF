import torch
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

from modeling_qwen3_asr_onnx import StatefulAudioEncoderWrapper, DiscreteAudioEncoder
from modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from export_config import MODEL_DIR, EXPORT_DIR

def verify_encoder():
    print("--- 正在验证 Audio Encoder ONNX 模型 ---")
    
    # 1. 加载原始 PyTorch 模型
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    encoder = full_model.thinker.audio_tower.eval()
    
    # 指定 ONNX 路径
    onnx_dir = Path(EXPORT_DIR) / "onnx"
    discrete_path = onnx_dir / "qwen3_asr_encoder_discrete.onnx"
    stateful_path = onnx_dir / "qwen3_asr_encoder_stateful.onnx"
    
    # 2. 准备测试数据 (B=1, T=128, F=128) - 使用 8 的倍化以对齐下采样
    dummy_mel = torch.randn(1, 128, 128)
    
    # 3. 运行 PyTorch 原生推理
    py_discrete = DiscreteAudioEncoder(encoder).eval()
    with torch.no_grad():
        expected_out = py_discrete(dummy_mel).numpy()
    
    # 4. 验证 Discrete ONNX
    print("\n[Test 1] 验证 Discrete ONNX...")
    sess_discrete = ort.InferenceSession(str(discrete_path))
    onnx_discrete_out = sess_discrete.run(None, {"mel": dummy_mel.numpy()})[0]
    
    mse_discrete = np.mean((expected_out - onnx_discrete_out) ** 2)
    print(f"Discrete ONNX MSE: {mse_discrete:.2e}")
    if mse_discrete < 1e-5:
        print("✅ Discrete 验证通过!")
    else:
        print("❌ Discrete 验证失败: 误差过大")

    # 5. 验证 Stateful ONNX (分块推理)
    print("\n[Test 2] 验证 Stateful ONNX (流式分块)...")
    sess_stateful = ort.InferenceSession(str(stateful_path))
    
    # 将 128 帧分为两块: 64 + 64
    chunk1 = dummy_mel[:, :64, :].numpy()
    chunk2 = dummy_mel[:, 64:, :].numpy()
    conv_state = np.zeros((1, 8, 128), dtype=np.float32)
    # 初始化 seq_offset
    seq_offset = np.array([0], dtype=np.int64)
    
    # 运行第一块
    out1, next_conv_state = sess_stateful.run(None, {
        "mel": chunk1,
        "conv_state": conv_state,
        "seq_offset": seq_offset
    })
    
    # 更新 seq_offset: 累加输出 token 数 (即 t - 1，因为我们已经切除了 overlap token)
    # out1 shape: [1, n_tokens, dim]
    seq_offset += out1.shape[1]
    
    # 运行第二块
    out2, _ = sess_stateful.run(None, {
        "mel": chunk2,
        "conv_state": next_conv_state,
        "seq_offset": seq_offset
    })
    
    # 拼接分块输出
    onnx_stateful_out = np.concatenate([out1, out2], axis=1)
    
    # 这里的对比比较微妙，因为卷积边界会有少许不同
    # 理想情况下，由于我们传递了 8 帧状态，结果应与全局推理高度一致
    # 但由于下采样对齐问题，可能需要检查中心部分
    
    print(f"原生输出 Shape: {expected_out.shape}")
    print(f"分块拼接 Shape: {onnx_stateful_out.shape}")
    
    # 如果形状一致，计算 MSE
    if expected_out.shape == onnx_stateful_out.shape:
        mse_stateful = np.mean((expected_out - onnx_stateful_out) ** 2)
        print(f"Stateful ONNX MSE (Full): {mse_stateful:.2e}")
        if mse_stateful < 1e-3: # 流式允许稍大的累积误差
             print("✅ Stateful 验证通过!")
        else:
             print("⚠️ Stateful 存在一定误差，请人工检查连续性")
    else:
        print("❌ Stateful 形状不匹配，流式下采样逻辑可能存在对齐偏移")

    # 6. 验证 Stateful ONNX (不规则分块 - 已对齐)
    # 注意：流式推理要求分块大小必须是下采样倍数 (8) 的整数倍，否则会导致卷积边缘对齐问题
    print("\n[Test 3] 验证 Stateful ONNX (不规则分块-对齐: [32, 40, 56])...") 
    chunk_sizes = [32, 40, 56] # Total 128
    conv_state = np.zeros((1, 8, 128), dtype=np.float32)
    seq_offset = np.array([0], dtype=np.int64)
    
    outputs = []
    current_idx = 0
    
    for size in chunk_sizes:
        chunk = dummy_mel[:, current_idx : current_idx + size, :].numpy()
        current_idx += size
        
        out, conv_state = sess_stateful.run(None, {
            "mel": chunk,
            "conv_state": conv_state,
            "seq_offset": seq_offset
        })
        outputs.append(out)
        seq_offset += out.shape[1]
        
    onnx_irregular_out = np.concatenate(outputs, axis=1)
    
    print(f"原生输出 Shape: {expected_out.shape}")
    print(f"不规则拼接 Shape: {onnx_irregular_out.shape}")
    
    if expected_out.shape == onnx_irregular_out.shape:
        mse_irregular = np.mean((expected_out - onnx_irregular_out) ** 2)
        print(f"Irregular Chunks MSE: {mse_irregular:.2e}")
        if mse_irregular < 1e-3:
             print("✅ 不规则分块(对齐)验证通过!")
        else:
             print("⚠️ 不规则分块(对齐)误差较大")
    else:
        print("❌ 不规则分块(对齐)形状不匹配")

    # 7. 随机压力测试 (Aligned)
    print("\n[Test 4] 运行随机压力测试 (Random Aligned Stress Test)...")
    np.random.seed(42)
    # 生成更长的序列进行测试 (确保是 8 的倍数)
    long_len = 512
    dummy_mel_long = torch.randn(1, long_len, 128)
    
    # 获取长序列的原生输出
    with torch.no_grad():
        expected_out_long = py_discrete(dummy_mel_long).numpy()
        
    # 随机生成分块 (必须是 8 的倍数)
    remaining = long_len
    chunks = []
    while remaining > 0:
        if remaining <= 16:
             chunks.append(remaining)
             break
        # 随机块大小 1*8 到 6*8 (8-48)
        size = np.random.randint(1, 7) * 8
        size = min(size, remaining)
        chunks.append(size)
        remaining -= size
        
    print(f"随机分块方案: {chunks}")
    
    conv_state = np.zeros((1, 8, 128), dtype=np.float32)
    seq_offset = np.array([0], dtype=np.int64)
    outputs_long = []
    current_idx = 0
    
    for size in chunks:
        chunk = dummy_mel_long[:, current_idx : current_idx + size, :].numpy()
        current_idx += size
        
        out, conv_state = sess_stateful.run(None, {
            "mel": chunk,
            "conv_state": conv_state,
            "seq_offset": seq_offset
        })
        outputs_long.append(out)
        seq_offset += out.shape[1]
        
    onnx_random_out = np.concatenate(outputs_long, axis=1)
    
    print(f"原生输出 Shape: {expected_out_long.shape}")
    print(f"随机拼接 Shape: {onnx_random_out.shape}")
    
    if expected_out_long.shape == onnx_random_out.shape:
        mse_random = np.mean((expected_out_long - onnx_random_out) ** 2)
        print(f"Random Stress MSE: {mse_random:.2e}")
        if mse_random < 1e-3:
             print("✅ 随机压力测试(对齐)通过!")
        else:
             print("⚠️ 随机压力测试(对齐)误差较大")
    else:
        print("❌ 随机压力测试(对齐)形状不匹配")

if __name__ == "__main__":
    verify_encoder()
