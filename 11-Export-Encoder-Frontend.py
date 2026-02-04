import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 将自定义模型目录添加到 sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

# 导入自定义组件
from modeling_qwen3_asr_onnx import EncoderConvFrontend
from export_config import MODEL_DIR, EXPORT_DIR
import torch.export

def export_frontend():
    """
    导出 Qwen3-ASR Audio Encoder 卷积前端为 ONNX。
    """
    print(f"--- 正在准备导出 Audio Encoder 卷积前端 (Frontend) ---")
    
    # 1. 加载模型
    print(f"正在加载原始模型权重: {MODEL_DIR}")
    from modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    
    try:
        full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        if hasattr(full_model, "thinker") and hasattr(full_model.thinker, "audio_tower"):
            encoder = full_model.thinker.audio_tower
        elif hasattr(full_model, "audio_tower"):
            encoder = full_model.audio_tower
        else:
            raise AttributeError("Model instance does not have 'audio_tower'")

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    output_dir = Path(EXPORT_DIR) / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 导出卷积前端 (Frontend)
    print("\n[Exporting] Conv Frontend...")
    frontend_wrapper = EncoderConvFrontend(encoder)
    frontend_wrapper.eval()
    
    frontend_path = output_dir / "qwen3_asr_encoder_frontend.fp32.onnx"
    
    # Dummy Data [B=1, T=123, F=128] (Force padding logic to be traced)
    dummy_mel_chunk = torch.randn(1, 123, 128)
    
    # 定义动态形状 (使用 Dynamo 推荐的 dynamic_shapes)
    dynamic_shapes = {
        "mel": {
            0: torch.export.Dim("batch", min=1, max=16),
            1: torch.export.Dim("n_frames", min=1, max=4096)
        }
    }

    try:
        torch.onnx.export(
            frontend_wrapper,
            (dummy_mel_chunk,),
            str(frontend_path),
            input_names=["mel"],
            output_names=["feat_out"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
        )
        print(f"✅ 卷积前端模型 (FP32) 已保存至: {frontend_path}")
    except Exception:
        import traceback
        print(f"❌ 导出前端模型 (FP32) 失败:")
        traceback.print_exc()

    # 3. 导出 FP16 版本 (针对 GPU DML 优化)
    print("\n[Exporting] Conv Frontend (FP16)...")
    try:
        # 将模型转为半精度
        frontend_wrapper.half()
        dummy_mel_half = dummy_mel_chunk.half()
        
        frontend_fp16_path = output_dir / "qwen3_asr_encoder_frontend.fp16.onnx"
        
        torch.onnx.export(
            frontend_wrapper,
            (dummy_mel_half,),
            str(frontend_fp16_path),
            input_names=["mel"],
            output_names=["feat_out"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
        )
        print(f"✅ 卷积前端模型 (FP16) 已保存至: {frontend_fp16_path}")
    except Exception:
        import traceback
        print(f"❌ 导出前端模型 (FP16) 失败:")
        traceback.print_exc()

if __name__ == "__main__":
    export_frontend()
