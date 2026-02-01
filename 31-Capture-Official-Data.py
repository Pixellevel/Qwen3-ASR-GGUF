import sys
from pathlib import Path
import torch
import numpy as np
import os
import shutil
import librosa

# 将自定义模型目录添加到 sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

# 如果 qwen_asr 本身不在路径中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 强制使用 transformers 后端，避免 vllm 干扰
os.environ["USE_VLLM"] = "0"

from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

# 配置
MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
AUDIO_PATH = "input.mp3"
CAPTURE_DIR = Path("capture")

def setup_capture():
    if CAPTURE_DIR.exists():
        shutil.rmtree(CAPTURE_DIR)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capture directory created: {CAPTURE_DIR}")

def save_npy(name, tensor):
    if tensor is None: return
    if isinstance(tensor, torch.Tensor):
        # 统一转为 float32 保存
        data = tensor.detach().cpu().to(torch.float32).numpy()
    else:
        data = np.array(tensor, dtype=np.float32)
    path = CAPTURE_DIR / f"{name}.npy"
    np.save(path, data)
    print(f"Saved: {path} | Shape: {data.shape}")

# Hooks
def encoder_pre_hook(module, args, kwargs):
    """
    args[0] 通常是 input_features
    """
    input_features = kwargs.get('input_features')
    if input_features is None and len(args) > 0:
        input_features = args[0]
    
    if input_features is not None:
        save_npy("01_mel_input", input_features)

def encoder_post_hook(module, input, output):
    """
    output 是 BaseModelOutput，包含 last_hidden_state
    """
    if hasattr(output, 'last_hidden_state'):
        save_npy("02_encoder_out", output.last_hidden_state)
    elif isinstance(output, torch.Tensor):
        save_npy("02_encoder_out", output)

def capture_official():
    setup_capture()
    
    # 1. 加载模型 (使用 float32 保证 CPU 精度)
    print("Loading model and processor...")
    # 注意：从 inference.py 看到 asr = Qwen3ASRModel.from_pretrained(...)
    asr = Qwen3ASRModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    
    # 真正的 Transformers 模型在 asr.model 中
    # Qwen3ASRForConditionalGeneration -> thinker -> audio_tower
    target_module = asr.model.thinker.audio_tower
    print(f"Target module identified: {type(target_module)}")
    
    # 2. 注册 Hooks
    target_module.register_forward_pre_hook(encoder_pre_hook, with_kwargs=True)
    target_module.register_forward_hook(encoder_post_hook)
    
    # 3. 运行推理 (会触发 hooks)
    print(f"Transcribing audio: {AUDIO_PATH}")
    # asr.transcribe 会处理音频加载和特征提取
    with torch.no_grad():
        results = asr.transcribe(
            audio=AUDIO_PATH,
            language=None,
            return_time_stamps=False
        )
    
    print("\n===== Transcription Result =====")
    for r in results:
        print(f"Text: {r.text}")

    print("\n✅ Official data capture complete!")

if __name__ == "__main__":
    capture_official()
