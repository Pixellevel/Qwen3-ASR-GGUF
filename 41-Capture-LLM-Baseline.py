import sys
from pathlib import Path
import torch
import numpy as np
import os
import shutil

# 将自定义模型目录添加到 sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

# 如果 qwen_asr 本身不在路径中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 强制使用 transformers 后端
os.environ["USE_VLLM"] = "0"

from qwen_asr import Qwen3ASRModel

# 配置
MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
AUDIO_PATH = "input.mp3"
CAPTURE_DIR = Path("capture_llm")

def setup_capture():
    if CAPTURE_DIR.exists():
        shutil.rmtree(CAPTURE_DIR)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capture directory created: {CAPTURE_DIR}")

def save_npy(name, tensor):
    if tensor is None: return
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().to(torch.float32).numpy()
    else:
        data = np.array(tensor, dtype=np.float32)
    path = CAPTURE_DIR / f"{name}.npy"
    np.save(path, data)
    print(f"Saved: {path} | Shape: {data.shape}")

# Hooks
captured = {"done": False}

def text_model_pre_hook(module, args, kwargs):
    """
    捕捉输入到 LLM Backbone 的 inputs_embeds
    """
    if captured["done"]: return
    
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is None and len(args) > 6: # 根据 forward 签名，inputs_embeds 是第 9 个参数
        # 稳妥起见，检查 kwargs 或特定的位置
        pass
    
    # 动态查找 inputs_embeds
    if inputs_embeds is not None:
        save_npy("llm_input_embeds", inputs_embeds)

def lm_head_post_hook(module, input, output):
    """
    捕捉 LM Head 输出的 Logits
    """
    if captured["done"]: return
    
    # output 是 [Batch, Seq, Vocab]
    # 我们只需要第一个生成位置的 logits，即序列末尾
    save_npy("llm_output_logits", output)
    captured["done"] = True

def capture_baseline():
    setup_capture()
    
    print("Loading model for LLM capture...")
    asr = Qwen3ASRModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    
    # Hook 点: 
    # asr.model -> Qwen3ASRForConditionalGeneration
    # asr.model.thinker -> Qwen3ASRThinkerForConditionalGeneration
    # asr.model.thinker.model -> Qwen3ASRThinkerTextModel (Backbone)
    # asr.model.thinker.lm_head -> Final Linear
    
    thinker = asr.model.thinker
    
    # 注入 Hooks
    thinker.model.register_forward_pre_hook(text_model_pre_hook, with_kwargs=True)
    thinker.lm_head.register_forward_hook(lm_head_post_hook)
    
    print(f"Running inference to capture LLM baseline...")
    with torch.no_grad():
        # 我们只推理一个短片段即可
        results = asr.transcribe(
            audio=AUDIO_PATH,
            language=None,
            return_time_stamps=False
        )
    
    print("\n✅ LLM baseline capture complete!")

if __name__ == "__main__":
    capture_baseline()
