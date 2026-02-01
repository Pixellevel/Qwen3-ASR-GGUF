import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.absolute()
EXPORT_DIR = PROJECT_ROOT / "model" # 假设 export_config 中的 EXPORT_DIR 指向这里

def verify_modular():
    print("--- [Script 14] 正在验证模块化 ONNX 联合推理 ---")
    
    data_dir = PROJECT_ROOT / "verify_data"
    mel_path = data_dir / "verify_mel.npy"
    gt_path = data_dir / "verify_ground_truth.npy"
    
    if not mel_path.exists() or not gt_path.exists():
        print(f"❌ 找不到测试数据，请先运行 13 号脚本。")
        return

    # 1. 加载测试数据
    mel = np.load(mel_path)
    ground_truth = np.load(gt_path)
    conv_state = np.load(data_dir / "verify_conv_state.npy")
    seq_offset = np.load(data_dir / "verify_seq_offset.npy")
    
    # 2. 加载 ONNX 模型
    onnx_dir = EXPORT_DIR / "onnx"
    frontend_path = onnx_dir / "qwen3_asr_encoder_frontend.onnx"
    backend_path = onnx_dir / "qwen3_asr_encoder_backend.int8.onnx"
    
    print(f"正在加载模型:\n- Frontend: {frontend_path}\n- Backend: {backend_path}")
    sess_frontend = ort.InferenceSession(str(frontend_path))
    sess_backend = ort.InferenceSession(str(backend_path))
    
    # 3. 运行模块化推理
    print("\n正在执行联合推理 (Frontend -> Backend)...")
    
    # Step A: Frontend
    feat_out = sess_frontend.run(None, {
        "mel": mel
    })[0]
    print(f"Frontend 成功，特征形状: {feat_out.shape}")
    
    # Step B: Backend
    modular_raw_out = sess_backend.run(None, {
        "feat_in": feat_out
    })[0]
    
    # 由于我们在 Backend ONNX 中移除了硬编码的切片 [:, 1:, :]，
    # 而官方 StatefulWrapper 基准数据是带切片的，这里手动模拟切片。
    modular_out = modular_raw_out[:, 1:, :]
    
    print(f"Backend 成功，原始输出形状: {modular_raw_out.shape}, 切片后: {modular_out.shape}")
    
    # 4. 误差对比
    mse = np.mean((ground_truth - modular_out) ** 2)
    max_diff = np.max(np.abs(ground_truth - modular_out))
    
    print(f"\n--- 验证结果 ---")
    print(f"Mean Squared Error (MSE): {mse:.2e}")
    print(f"Max Absolute Difference: {max_diff:.2e}")
    
    if mse < 1e-5:
        print("✅ 模块化联合推理验证通过！与 PyTorch 结果高度一致。")
    else:
        print("❌ 验证失败：误差超出范围，请检查导出逻辑。")

if __name__ == "__main__":
    verify_modular()
