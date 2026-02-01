import numpy as np
import onnxruntime as ort
from pathlib import Path

# 配置
ONNX_PATH = "model/onnx/qwen3_asr_encoder_discrete.onnx"
CAPTURE_DIR = Path("capture")

def verify_capture():
    print(f"--- 正在使用捕获的数据验证 ONNX 模型 ---")
    
    # 1. 检查数据
    mel_path = CAPTURE_DIR / "01_mel_input.npy"
    ref_path = CAPTURE_DIR / "02_encoder_out.npy"
    
    if not mel_path.exists() or not ref_path.exists():
        print(f"❌ 找不到捕获的数据文件。请先运行 31-Capture-Official-Data.py")
        return

    mel_in = np.load(mel_path).astype(np.float32)
    ref_out = np.load(ref_path).astype(np.float32)
    
    # 如果缺少 batch 维度，补上
    if mel_in.ndim == 2:
        mel_in = np.expand_dims(mel_in, axis=0)
    if ref_out.ndim == 2:
        ref_out = np.expand_dims(ref_out, axis=0)

    # 处理维度：官方 Processor 输出可能是 [B, Mel, T] (1, 128, 3000)
    # 我们的 ONNX 包装期望 [B, T, Mel] (1, 3000, 128)
    if mel_in.shape[2] != 128 and mel_in.shape[1] == 128:
        print(f"Detecting [B, Mel, T] format, transposing to [B, T, Mel]...")
        mel_in = np.transpose(mel_in, (0, 2, 1))
    
    print(f"Input Mel Shape: {mel_in.shape}")
    print(f"Reference Output Shape: {ref_out.shape}")
    
    # 2. 加载 ONNX
    print(f"Loading ONNX model: {ONNX_PATH}")
    options = ort.SessionOptions()
    sess = ort.InferenceSession(ONNX_PATH, options, providers=["CPUExecutionProvider"])
    
    # 3. 运行 ONNX
    # 注意：Discrete 模型的输入名称是 "mel"
    print("Running ONNX inference...")
    onnx_out = sess.run(None, {"mel": mel_in})[0]
    
    print(f"ONNX Output Shape: {onnx_out.shape}")
    
    # 4. 比较
    if onnx_out.shape != ref_out.shape:
        print(f"❌ 形状不匹配! ONNX: {onnx_out.shape}, Ref: {ref_out.shape}")
        # 尝试检查是否只是末尾 padding 或 下采样对齐问题
        return
        
    mse = np.mean((onnx_out - ref_out) ** 2)
    mae = np.max(np.abs(onnx_out - ref_out))
    
    print(f"\n结果对比:")
    print(f"Mean Squared Error (MSE): {mse:.2e}")
    print(f"Max Absolute Error (MAE): {mae:.2e}")
    
    if mse < 1e-4:
        print("\n✅ Discrete ONNX 与官方模型输出一致性验证通过!")
    else:
        print("\n⚠️ Discrete 存在显著差异")

    # 5. 验证 Stateful (流式)
    STATEFUL_ONNX = "model/onnx/qwen3_asr_encoder_stateful.onnx"
    if not Path(STATEFUL_ONNX).exists():
         return

    print(f"\n--- 正在验证 Stateful ONNX (Seamless vs Official Seamy) ---")
    sess_st = ort.InferenceSession(STATEFUL_ONNX, options, providers=["CPUExecutionProvider"])
    
    # 模拟流式推理
    conv_state = np.zeros((1, 8, 128), dtype=np.float32)
    seq_offset = np.array([0], dtype=np.int64)
    chunk_size = 16 # 使用 16 帧的小块进行流式演练
    
    all_chunks = []
    # 循环时我们直接从 mel_in 中切片
    for i in range(0, mel_in.shape[1], chunk_size):
        chunk = mel_in[:, i:i+chunk_size, :]
        if chunk.shape[1] < chunk_size:
            # 补齐到 8 的倍数以防卷积层报错 (Stateful 包装里有 cat 逻辑)
            # 但这里我们主要看 MSE
            pass
            
        ort_inputs = {
            "mel": chunk,
            "conv_state": conv_state,
            "seq_offset": seq_offset
        }
        out, next_conv_state = sess_st.run(None, ort_inputs)
        all_chunks.append(out)
        
        conv_state = next_conv_state
        seq_offset = seq_offset + out.shape[1]
        
    stateful_out = np.concatenate(all_chunks, axis=1)
    print(f"Stateful Output Shape: {stateful_out.shape}")
    
    # 因为采样 8 的倍数，长度应当一致
    if stateful_out.shape == ref_out.shape:
        mse_st = np.mean((stateful_out - ref_out) ** 2)
        print(f"Stateful vs Official MSE: {mse_st:.2e}")
        if mse_st < 1e-3:
            print("✅ Stateful 结果在可接受范围内 (考虑到流式连续性与官方分块差异)")
        else:
            print("⚠️ Stateful 差异较大，这可能是由于官方模型的分块缝隙导致的")
    else:
        print(f"Stateful 形状不匹配: {stateful_out.shape} vs {ref_out.shape}")

if __name__ == "__main__":
    verify_capture()
