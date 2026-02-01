import torch
import numpy as np
import os
import sys
from pathlib import Path
from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast

# 路径设置
PROJECT_ROOT = Path(__file__).parent.absolute()
MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

def instrument_official_model():
    print("--- [Investigation] 正在插装官方模型以抓取真实内部状态 ---")
    
    # 1. 加载模型与 Processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_DIR)
    processor = Qwen3ASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).eval()
    
    # 2. 准备真实音频输入 (test.mp3 对应的 Mel)
    import librosa
    audio_path = os.path.join(PROJECT_ROOT, "test.mp3")
    if not os.path.exists(audio_path):
        print(f"❌ 找不到 test.mp3")
        return
        
    audio, _ = librosa.load(audio_path, sr=16000)
    # 取前 10 秒进行分析，避免过大，但要足以跨越多个窗口
    audio_clip = audio[:16000 * 10] 
    inputs = processor(text="语音转录：", audio=audio_clip, return_tensors="pt")
    
    # 3. 插装 Hook：抓取 audio_tower 的最终输出
    captured_data = {}
    
    def hook_fn(module, input, output):
        # output is Qwen3ASRAudioEncoderOutputWithState or similar
        # contain .last_hidden_state
        print(f"Hook 触发！抓取到输出形状: {output.last_hidden_state.shape}")
        captured_data['audio_features'] = output.last_hidden_state.detach().cpu().numpy()

    # 找到 audio_tower 并挂载 hook
    # 结构: model (Qwen3ASRForConditionalGeneration) -> thinker -> audio_tower
    handle = model.thinker.audio_tower.register_forward_hook(hook_fn)
    
    # 4. 执行推理
    print("开始正式推理...")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=10
        )
    
    handle.remove()
    
    # 5. 保存真实特征
    if 'audio_features' in captured_data:
        save_path = PROJECT_ROOT / "verify_data" / "official_internal_features.npy"
        np.save(save_path, captured_data['audio_features'])
        np.save(PROJECT_ROOT / "verify_data" / "official_mel.npy", inputs['input_features'].numpy())
        print(f"✅ 官方模型内部特征已保存至: {save_path}")
        print(f"特征形状: {captured_data['audio_features'].shape}")
    else:
        print("❌ 未抓取到特征")

if __name__ == "__main__":
    instrument_official_model()
