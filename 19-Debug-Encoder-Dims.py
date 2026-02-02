
import os
import sys
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import librosa
from pathlib import Path
from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast

# 添加本地库路径
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

HF_MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
FRONTEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_frontend.onnx")
BACKEND_ONNX_PATH = os.path.join(PROJECT_ROOT, "model", "onnx", "qwen3_asr_encoder_backend.int8.onnx")

def test_dimensions(audio_path):
    print(f"--- Testing Dimensions for {audio_path} ---")
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return

    # 1. Load Processor & Feature Extraction
    print("Loading Processor...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL_DIR)
    tokenizer = Qwen2TokenizerFast.from_pretrained(HF_MODEL_DIR, fix_mistral_regex=True)
    processor = Qwen3ASRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    audio, _ = librosa.load(audio_path, sr=16000)
    print(f"Audio Length: {len(audio)} samples ({len(audio)/16000:.2f}s)")
    
    inputs = processor(text="test", audio=audio, return_tensors="pt")
    mel = inputs.input_features.numpy()
    if mel.shape[1] == 128:
        mel = mel.transpose(0, 2, 1) # [1, 128, T] -> [1, T, 128] ? 
        # Wait, Qwen3 expects [B, T, D] for frontend input in ONNX wrapper?
        # Let's check modeling_qwen3_asr_onnx.py: "forward(self, mel): b, t, d = mel.size()"
        # So it expects [Batch, Time, Dim].
        # WhisperFE output is usually [1, 128, 3000].
        # So we need transpose (0, 2, 1) -> [1, 3000, 128].
        pass
        
    print(f"Mel Input Shape: {mel.shape}")
    
    # 2. Run Frontend ONNX
    print("Running Frontend ONNX...")
    ts_fe_out = None
    try:
        sess_fe = ort.InferenceSession(FRONTEND_ONNX_PATH, providers=['CPUExecutionProvider'])
        # Check input name
        input_name = sess_fe.get_inputs()[0].name
        print(f"Frontend Input Name: {input_name}")
        
        feat_out = sess_fe.run(None, {input_name: mel})[0]
        print(f"✅ Frontend Output Shape: {feat_out.shape}")
        ts_fe_out = feat_out
    except Exception as e:
        print(f"❌ Frontend execution failed: {e}")
        return

    # 3. Inspect Frontend Output
    # Check if dimensions look like [B, T, D] or [B, D, T]
    B, D2, D3 = feat_out.shape
    print(f"Shape Analysis: B={B}, D2={D2}, D3={D3}")
    
    # Expected: [1, n_tokens, 1024]
    if D3 == 1024:
        print("Looks like [B, T, 1024]. Correct.")
    elif D2 == 1024:
        print("Looks like [B, 1024, T]. TRANSPOSED/INCORRECT!")
    else:
        print(f"Unknown shape format. Neither D2 nor D3 is 1024.")

    # 4. Run Backend ONNX
    print("Running Backend ONNX...")
    try:
        sess_be = ort.InferenceSession(BACKEND_ONNX_PATH, providers=['CPUExecutionProvider'])
        input_name = sess_be.get_inputs()[0].name
        print(f"Backend Input Name: {input_name}")
        # Try running
        res = sess_be.run(None, {input_name: ts_fe_out})[0]
        print(f"✅ Backend execution successful. Output: {res.shape}")
    except Exception as e:
        print(f"❌ Backend execution failed: {e}")

if __name__ == "__main__":
    # Prioritize test40.mp3 if exists, else test.mp3, else input.mp3
    target = "test40.mp3"
    if not os.path.exists(target):
         target = "out.mp3" # The one that failed in user report
    if not os.path.exists(target):
         target = "test.mp3"
         
    test_dimensions(target)
