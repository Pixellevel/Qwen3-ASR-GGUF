import os
import sys
import json
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from export_config import MODEL_DIR, EXPORT_DIR

def extract_encoder_weights():
    print(f"Loading Qwen3-ASR model from: {MODEL_DIR}")
    
    # Load the full model to ensure correct class initialization
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_DIR, trust_remote_code=True, device_map="cpu")
    
    # Access the audio encoder
    # Structure from modeling_qwen3_asr.py:
    # Qwen3ASRForConditionalGeneration -> thinker (Qwen3ASRThinkerForConditionalGeneration) -> audio_tower (Qwen3ASRAudioEncoder)
    
    if not hasattr(model, "thinker"):
         raise ValueError("Model does not have 'thinker' attribute.")
         
    if not hasattr(model.thinker, "audio_tower"):
         raise ValueError("Model.thinker does not have 'audio_tower' attribute.")
         
    audio_encoder = model.thinker.audio_tower
    print("Found audio_tower (Audio Encoder).")
    
    output_dir = Path(EXPORT_DIR) / "encoder_transformer_hf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting Audio Encoder weights to: {output_dir}")
    
    # 1. Save Config
    encoder_config = audio_encoder.config
    config_dict = encoder_config.to_dict()
    # Add architecture for easy loading if we wrap it, or just for reference
    config_dict["architectures"] = ["Qwen3ASRAudioEncoder"]
    config_dict["model_type"] = "qwen3_asr_audio_encoder"
    
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print("Saved config.json")
    
    # 2. Extract Weights
    # We can just take the state_dict of the submodule
    # This automatically removes the "thinker.audio_tower." prefix!
    state_dict = audio_encoder.state_dict()
    
    print(f"Extracted {len(state_dict)} tensors.")
    
    # Save weights
    save_file(state_dict, output_dir / "model.safetensors")
    print("Saved model.safetensors")
    
    print("\n✅ Extraction complete!")

if __name__ == "__main__":
    extract_encoder_weights()
