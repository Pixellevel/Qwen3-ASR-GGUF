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
from modeling_qwen3_asr_onnx import StatefulAudioEncoderWrapper, DiscreteAudioEncoder
from modeling_qwen3_asr import Qwen3ASRAudioEncoder
from configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from export_config import MODEL_DIR, EXPORT_DIR

def export_audio_encoder():
    """
    导出 Qwen3-ASR Audio Encoder 为 ONNX。
    """
    print(f"--- 正在准备导出 Audio Encoder ---")
    
    # 1. 加载配置和权重
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
            print(f"DEBUG: Available attributes: {dir(full_model)}")
            if hasattr(full_model, "thinker"):
                print(f"DEBUG: Thinker attributes: {dir(full_model.thinker)}")
            raise AttributeError("Model instance does not have 'audio_tower' (direct or in thinker)")

        config = encoder.config
        # 强制使用 eager attention 以便 ONNX 追踪
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        print(f"✅ 成功加载 Encoder。层数: {config.encoder_layers}, 维度: {config.d_model}, Attn: {getattr(config, '_attn_implementation', 'n/a')}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    output_dir = Path(EXPORT_DIR) / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 导出全量 (Discrete) 版本
    print("\n[Stage 1/2] 正在导出全量 ONNX 模型 (Discrete)...")
    discrete_wrapper = DiscreteAudioEncoder(encoder)
    discrete_wrapper.eval()
    
    # 准备 Dummy Data [B=1, T=100, F=128]
    dummy_mel = torch.randn(1, 100, 128)
    discrete_path = output_dir / "qwen3_asr_encoder_discrete.onnx"
    
    try:
        torch.onnx.export(
            discrete_wrapper,
            (dummy_mel,),
            str(discrete_path),
            input_names=["mel"],
            output_names=["hidden_states"],
            dynamic_axes={
                "mel": {1: "n_frames"},
                "hidden_states": {1: "n_tokens"}
            },
            opset_version=17,
            do_constant_folding=True,
            # 关键：根据经验，禁用 Dynamo 以提高兼容性
            **({"dynamo": False} if hasattr(torch.onnx, "export") else {})
        )
        print(f"✅ 全量模型已保存至: {discrete_path}")
    except Exception:
        import traceback
        print(f"❌ 导出全量模型失败:")
        traceback.print_exc()

    # 3. 导出流式 (Stateful) 版本
    print("\n[Stage 2/2] 正在导出流式 ONNX 模型 (Stateful)...")
    stateful_wrapper = StatefulAudioEncoderWrapper(encoder)
    stateful_wrapper.eval()
    
    dummy_mel_chunk = torch.randn(1, 50, 128)
    dummy_conv_state = torch.randn(1, 8, 128)
    # 新增: seq_offset 输入 (int64 scalar)
    dummy_seq_offset = torch.tensor([0], dtype=torch.int64)
    
    stateful_path = output_dir / "qwen3_asr_encoder_stateful.onnx"
    
    try:
        torch.onnx.export(
            stateful_wrapper,
            (dummy_mel_chunk, dummy_conv_state, dummy_seq_offset),
            str(stateful_path),
            input_names=["mel", "conv_state", "seq_offset"],
            output_names=["hidden_states", "next_conv_state"],
            dynamic_axes={
                "mel": {1: "n_frames"},
                "hidden_states": {1: "n_tokens"}
            },
            opset_version=17,
            do_constant_folding=True,
            # 关键：根据经验，禁用 Dynamo 以提高兼容性
            **({"dynamo": False} if hasattr(torch.onnx, "export") else {})
        )
        print(f"✅ 流式模型已保存至: {stateful_path}")
    except Exception:
        import traceback
        print(f"❌ 导出流式模型失败:")
        traceback.print_exc()

if __name__ == "__main__":
    export_audio_encoder()
