import torch
import numpy as np
import sys
import os
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

from modeling_qwen3_asr_onnx import StatefulAudioEncoderWrapper
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
from export_config import MODEL_DIR

def capture_ground_truth():
    print("--- [Script 13] 正在捕捉官方模型基准数据 (30秒音频) ---")
    
    # 1. 加载模型
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    encoder = full_model.thinker.audio_tower.eval()
    
    # 2. 准备输入 (30秒 = 3000 帧)
    torch.manual_seed(42)
    dummy_mel = torch.randn(1, 3000, 128)
    dummy_conv_state = torch.randn(1, 8, 128) 
    dummy_seq_offset = torch.tensor([0], dtype=torch.int64)
    
    # 3. 运行推理
    print("运行 PyTorch Stateful 推理...")
    wrapper = StatefulAudioEncoderWrapper(encoder)
    with torch.no_grad():
        # 注意：StatefulWrapper 返回 (hidden_states, next_conv_state)
        ground_truth, _ = wrapper(dummy_mel, dummy_conv_state, dummy_seq_offset)
        ground_truth = ground_truth.numpy()
    
    # 4. 保存数据
    data_dir = PROJECT_ROOT / "verify_data"
    data_dir.mkdir(exist_ok=True)
    
    np.save(data_dir / "verify_mel.npy", dummy_mel.numpy())
    np.save(data_dir / "verify_conv_state.npy", dummy_conv_state.numpy())
    np.save(data_dir / "verify_seq_offset.npy", dummy_seq_offset.numpy())
    np.save(data_dir / "verify_ground_truth.npy", ground_truth)
    
    print(f"✅ 数据已保存至 {data_dir}")
    print(f"输入形状: {dummy_mel.shape}")
    print(f"输出形状: {ground_truth.shape}")

if __name__ == "__main__":
    capture_ground_truth()
