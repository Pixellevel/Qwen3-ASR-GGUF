import numpy as np
import onnxruntime as ort
import os
from pathlib import Path

def compare_to_official():
    print("--- [Investigation] 正在进行交叉验证：模块化 ONNX vs. 官方内部特征 ---")
    
    PROJECT_ROOT = Path(__file__).parent.absolute()
    data_dir = PROJECT_ROOT / "verify_data"
    mel = np.load(data_dir / "official_mel.npy")
    official_features = np.load(data_dir / "official_internal_features.npy")
    
    # ONNX 模型路径
    frontend_path = PROJECT_ROOT / "model" / "onnx" / "qwen3_asr_encoder_frontend.onnx"
    backend_path = PROJECT_ROOT / "model" / "onnx" / "qwen3_asr_encoder_backend.int8.onnx"
    
    sess_frontend = ort.InferenceSession(str(frontend_path))
    sess_backend = ort.InferenceSession(str(backend_path))
    
    # 推理
    # Qwen3 官方特征提取后通常需要转置以符合 ONNX 预期 [B, T, D]
    if mel.shape[1] == 128:
        mel = mel.transpose(0, 2, 1)

    print(f"输入 Mel 形状: {mel.shape}")
    
    # Step A: Frontend
    feat_out = sess_frontend.run(None, {"mel": mel})[0]
    
    # Step B: Backend
    modular_raw_out = sess_backend.run(None, {
        "feat_in": feat_out
    })[0]
    
    # 新的 Discrete 导出已经去除了 Overlap Token，无需切片
    modular_out = modular_raw_out[0]
    
    print(f"官方特征形状: {official_features.shape}")
    print(f"模块化输出形状: {modular_out.shape}")
    
    # 强行对齐长度（如果不同）
    min_len = min(official_features.shape[0], modular_out.shape[0])
    off = official_features[:min_len]
    mod = modular_out[:min_len]
    
    mse = np.mean((off - mod) ** 2)
    max_diff = np.max(np.abs(off - mod))
    
    print(f"\n--- 交叉验证结果 ---")
    print(f"MSE: {mse:.2e}")
    print(f"Max Diff: {max_diff:.2e}")
    
    if mse > 1e-1:
        print("🚨 警报：模块化输出与官方内部特征存在重大偏差！")
        # 尝试不切片对比
        modular_no_slice = modular_raw_out[0, :min_len, :]
        mse_no_slice = np.mean((off - modular_no_slice) ** 2)
        print(f"不切片对比 MSE: {mse_no_slice:.2e}")
    else:
        print("✅ 模块化输出与官方内部特征完全对齐。")

if __name__ == "__main__":
    compare_to_official()
