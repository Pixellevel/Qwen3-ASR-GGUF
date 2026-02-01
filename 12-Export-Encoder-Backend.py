import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 将自定义模型目录添加到 sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

# 导入自定义组件
from modeling_qwen3_asr_onnx import EncoderTransformerBackend
from export_config import MODEL_DIR, EXPORT_DIR

def export_backend_int8():
    """
    导出 Qwen3-ASR Audio Encoder Transformer 后端并进行 INT8 量化。
    """
    print(f"--- 正在准备导出 Audio Encoder Transformer 后端 (INT8 量化) ---")
    
    # 1. 加载模型
    print(f"正在加载原始模型权重: {MODEL_DIR}")
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
    
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

        config = encoder.config
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    output_dir = Path(EXPORT_DIR) / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 导出 FP32 中间模型
    print("\n[Stage 1/2] 导出 FP32 模型...")
    backend_wrapper = EncoderTransformerBackend(encoder)
    backend_wrapper.eval()
    
    dummy_feat_in = torch.randn(1, 13, encoder.config.d_model)
    
    fp32_path = output_dir / "qwen3_asr_encoder_backend.fp32.onnx"
    # 如果已经存在全量的 backend 模型，可以直接量化它。
    # 但为了脚本独立性，我们这里先导出一个临时的。
    
    try:
        torch.onnx.export(
            backend_wrapper,
            (dummy_feat_in,),
            str(fp32_path),
            input_names=["feat_in"],
            output_names=["hidden_states"],
            dynamic_axes={
                "feat_in": {1: "n_tokens"},
                "hidden_states": {1: "n_tokens"}
            },
            opset_version=17,
            do_constant_folding=True,
            **({"dynamo": False} if hasattr(torch.onnx, "export") else {})
        )
        print(f"✅ FP32 模型已导出。")
    except Exception:
        import traceback
        print(f"❌ 导出 FP32 模型失败:")
        traceback.print_exc()
        return

    # 3. 动态量化为 INT8
    print("\n[Stage 2/2] 正在进行 INT8 动态量化 (针对 MatMul)...")
    int8_path = output_dir / "qwen3_asr_encoder_backend.int8.onnx"
    
    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        print(f"✅ INT8 量化模型已保存至: {int8_path}")
        
        # 可选：删除临时 FP32 文件
        # os.remove(fp32_path)
    except Exception as e:
        print(f"❌ 量化失败: {e}")

if __name__ == "__main__":
    export_backend_int8()
