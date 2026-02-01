import os
import sys
import torch
import numpy as np
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "qwen_asr_gguf" / "qwen3_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

# 导入
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast

MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
AUDIO_PATH = "test.mp3"

def main():
    print(f"--- 提取 {AUDIO_PATH} 的 Mel 特征 ---")
    
    # 1. 初始化 Processor
    # Qwen3ASRProcessor 内部使用 WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_DIR)
    processor = Qwen3ASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # 2. 加载音频
    import librosa
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    
    # 3. 提取特征
    inputs = processor(text="语音转录：", audio=audio, return_tensors="pt")
    
    # input_features shape: [1, 128, T_mel]
    mel = inputs.input_features
    print(f"Mel Feature Shape: {mel.shape}")
    
    # 保存 Mel 特征供后续 ONNX 使用
    np.save("test_mel.npy", mel.numpy())
    print("✅ Mel 特征已保存至 test_mel.npy")

if __name__ == "__main__":
    main()
