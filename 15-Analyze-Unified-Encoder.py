import numpy as np
import onnxruntime as ort
import os
from pathlib import Path

def analyze_unified_encoder():
    print("--- [Investigation] 正在分析合体 Encoder 精度与 Token 数量 ---")
    
    PROJECT_ROOT = Path(__file__).parent.absolute()
    data_dir = PROJECT_ROOT / "verify_data"
    
    # 1. 加载官方“真金”数据
    if not os.path.exists(data_dir / "official_internal_features.npy"):
        print("❌ 找不到 official_internal_features.npy，请先运行 13b 脚本。")
        return
    mel = np.load(data_dir / "official_mel.npy")
    official_features = np.load(data_dir / "official_internal_features.npy")
    
    # 2. 加载合体 ONNX 模型
    unified_path = PROJECT_ROOT / "model" / "onnx" / "qwen3_asr_encoder_discrete_all.onnx"
    if not unified_path.exists():
        print(f"❌ 找不到合体模型: {unified_path}")
        return
        
    sess_unified = ort.InferenceSession(str(unified_path))
    
    # 3. 运行合体模型推理
    if mel.shape[1] == 128:
        mel = mel.transpose(0, 2, 1)
        
    unified_out = sess_unified.run(None, {"mel": mel})[0][0]
    
    print(f"\n--- 长度对比 (10秒音频) ---")
    print(f"官方 Token 数量: {official_features.shape[0]}")
    print(f"合体 ONNX Token 数量: {unified_out.shape[0]}")
    
    # 4. 精度对比
    min_len = min(official_features.shape[0], unified_out.shape[0])
    mse = np.mean((official_features[:min_len] - unified_out[:min_len]) ** 2)
    max_diff = np.max(np.abs(official_features[:min_len] - unified_out[:min_len]))
    
    print(f"\n--- 精度对比 (合体 ONNX vs 官方) ---")
    print(f"MSE: {mse:.2e}")
    print(f"Max Diff: {max_diff:.2e}")
    
    # 5. 回顾模块化数据 (如果存在)
    # 之前调查显示模块化产生 125 tokens，这里再次确认为何合体能产生 130 tokens
    frontend_path = PROJECT_ROOT / "model" / "onnx" / "qwen3_asr_encoder_frontend.onnx"
    backend_path = PROJECT_ROOT / "model" / "onnx" / "qwen3_asr_encoder_backend.int8.onnx"
    
    if frontend_path.exists() and backend_path.exists():
        sess_fe = ort.InferenceSession(str(frontend_path))
        sess_be = ort.InferenceSession(str(backend_path))
        
        feat_out, _ = sess_fe.run(None, {"mel": mel, "conv_state": np.zeros((1, 8, 128), dtype=np.float32)})
        modular_raw_out = sess_be.run(None, {"feat_in": feat_out, "seq_offset": np.array([0], dtype=np.int64)})[0]
        modular_out = modular_raw_out[0, 1:, :] # 模拟 transcribe 逻辑
        
        print(f"\n--- 模块化对比 ---")
        print(f"模块化 Token 数量: {modular_out.shape[0]}")
        mse_mod = np.mean((official_features[:min(130, modular_out.shape[0])] - modular_out[:min(130, modular_out.shape[0])]) ** 2)
        print(f"模块化 vs 官方 MSE: {mse_mod:.2e}")

if __name__ == "__main__":
    analyze_unified_encoder()
