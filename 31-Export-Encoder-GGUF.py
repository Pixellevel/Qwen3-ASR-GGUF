import os
import sys
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter, GGUFWriter, GGMLQuantizationType

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
from export_config import MODEL_DIR, EXPORT_DIR

def export_encoder_gguf():
    print(f"Loading Qwen3-ASR Audio Encoder from: {MODEL_DIR}")
    
    # Load just the encoder part if possible, or extract it from the full model
    # To avoid loading the whole 14GB model if we only need the encoder (unless it's integrated)
    # The user's code suggests loading `Qwen3ASRForConditionalGeneration` usually.
    # checking imports in user script: from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    # But we can try loading just the encoder with the config.
    
    # However, to get the weights, we probably need the full model checkpoint unless we have split checkpoints.
    # Assuming standard HF single folder.
    
    # Let's try loading Qwen3ASRForConditionalGeneration (safest)
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_DIR, trust_remote_code=True, device_map="cpu")
    encoder = model.model.audio_encoder # Check path: model(Qwen3ASR) -> model(Qwen3ASRThinker) -> audio_encoder(Qwen3ASRAudioEncoder)
    # Actually Qwen3ASRForConditionalGeneration usually wraps Qwen3ASRThinker as `model`?
    # Let's check `modeling_qwen3_asr.py` class structure briefly/recall it.
    # Qwen3ASRForConditionalGeneration has `model` which is `Qwen3ASRAudioEncoder`? No, `thinker`.
    # `Qwen3ASRForConditionalGeneration` inherits `Qwen3ASRPreTrainedModel`.
    # It usually has `model = Qwen3ASRThinker(...)` (based on typical composition) or `model` is `Qwen3ASRThinker`.
    # Based on `21-Export-ASR-LLM.py`: `model.config.thinker_config` implies `model` is the top level.
    # Let's assume standard `model.audio_encoder` or `model.thinker.audio_encoder`.
    # Looking at `configuration_qwen3_asr.py`: `Qwen3ASRThinkerConfig` has `audio_config`.
    # So `model.thinker` likely has `audio_encoder`.
    
    # Let's inspect `model` dynamically if strict path is unknown, or inspect `modeling_qwen3_asr.py` again?
    # `modeling_qwen3_asr.py` showed `Qwen3ASRAudioEncoder` exists.
    # Let's assume `model.model.audio_encoder` or `model.audio_encoder` for now.
    # Actually, if I look at `21-Export-ASR-LLM.py`: `model.config.thinker_config`, so `model` has a `thinker`?
    # Or `model` IS the thinker wrapper?
    # Let's try to access `model.audio_encoder` directly if it's exposed, or traverse.
    
    # Correct path likely: `model.model.audio_encoder` or `model.thinker.audio_encoder`.
    # Wait, `Qwen3ASRForConditionalGeneration` usually has `model` attribute which is the inner model.
    # Let's look at `modeling_qwen3_asr.py` again? No, I'll just use traversal or print.
    # Actually simpler: load `Qwen3ASRForConditionalGeneration` and search for `Qwen3ASRAudioEncoder` module.
    
    audio_encoder = None
    for name, module in model.named_modules():
        if isinstance(module, Qwen3ASRAudioEncoder):
            audio_encoder = module
            print(f"Found Audio Encoder at: {name}")
            break
            
    if audio_encoder is None:
        raise ValueError("Could not find Qwen3ASRAudioEncoder in the model.")

    output_path = Path(EXPORT_DIR) / "qwen3_encoder.gguf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to: {output_path}")
    
    gguf_writer = GGUFWriter(output_path, "bert")
    
    # Configuration
    config = audio_encoder.config
    
    # BERT Params
    # bert.context_length
    # bert.embedding_length (d_model)
    # bert.feed_forward_length (encoder_ffn_dim)
    # bert.attention.head_count
    # bert.attention.layer_count
    
    gguf_writer.add_name("Qwen3-ASR-AudioEncoder-Transformer")
    gguf_writer.add_context_length(config.max_source_positions) # 1500
    gguf_writer.add_embedding_length(config.d_model) # 1280
    gguf_writer.add_feed_forward_length(config.encoder_ffn_dim) # 5120
    gguf_writer.add_head_count(config.encoder_attention_heads) # 20
    gguf_writer.add_block_count(config.encoder_layers) # 32
    gguf_writer.add_layer_norm_eps(1e-6) # Default LayerNorm eps, checking Qwen... 
    # Qwen uses nn.LayerNorm(self.embed_dim). Default eps is 1e-5.
    # `modeling_qwen3_asr.py` doesn't pass eps to `self_attn_layer_norm = nn.LayerNorm(self.embed_dim)`.
    # So it uses pytorch default 1e-5.
    gguf_writer.add_layer_norm_eps(1e-5)

    # Tensor Mapping
    state_dict = audio_encoder.state_dict()
    
    # Mapping rules
    # blk.{i}.attn_q.weight
    # blk.{i}.attn_k.weight
    # blk.{i}.attn_v.weight
    # blk.{i}.attn_output.weight
    # blk.{i}.attn_output_norm.weight (self_attn_layer_norm)
    # blk.{i}.attn_output_norm.bias
    # blk.{i}.ffn_up.weight (fc1)
    # blk.{i}.ffn_up.bias
    # blk.{i}.ffn_down.weight (fc2)
    # blk.{i}.ffn_down.bias
    # blk.{i}.layer_output_norm.weight (final_layer_norm)
    # blk.{i}.layer_output_norm.bias
    
    # Positional Embedding
    # model.positional_embedding.positional_embedding -> pos_embd.weight
    # Note: Qwen3 uses fixed Sinusoidal. BERT uses learned. We verify shape.
    # Qwen pos embd: [max_source_positions, d_model]
    
    mapping = {
        "self_attn.q_proj.weight": "blk.{}.attn_q.weight",
        "self_attn.q_proj.bias": "blk.{}.attn_q.bias",
        "self_attn.k_proj.weight": "blk.{}.attn_k.weight",
        "self_attn.k_proj.bias": "blk.{}.attn_k.bias",
        "self_attn.v_proj.weight": "blk.{}.attn_v.weight",
        "self_attn.v_proj.bias": "blk.{}.attn_v.bias",
        "self_attn.out_proj.weight": "blk.{}.attn_output.weight",
        "self_attn.out_proj.bias": "blk.{}.attn_output.bias",
        
        "self_attn_layer_norm.weight": "blk.{}.attn_output_norm.weight",
        "self_attn_layer_norm.bias": "blk.{}.attn_output_norm.bias",
        
        "fc1.weight": "blk.{}.ffn_up.weight",
        "fc1.bias": "blk.{}.ffn_up.bias",
        
        "fc2.weight": "blk.{}.ffn_down.weight",
        "fc2.bias": "blk.{}.ffn_down.bias",
        
        "final_layer_norm.weight": "blk.{}.layer_output_norm.weight",
        "final_layer_norm.bias": "blk.{}.layer_output_norm.bias",
    }
    
    # Count of tensors added
    added_count = 0
    
    for key, tensor in state_dict.items():
        # Handle Layers
        if key.startswith("layers."):
            parts = key.split(".")
            layer_idx = parts[1]
            suffix = ".".join(parts[2:])
            
            if suffix in mapping:
                new_key = mapping[suffix].format(layer_idx)
                # Convert to standard format usually expected (numpy, FP32 or FP16)
                # We'll save as F16 to save space if it's feasible, or F32.
                # GGUFWriter handles it.
                data = tensor.numpy().astype(np.float32) # Convert to F32 for safety
                gguf_writer.add_tensor(new_key, data)
                added_count += 1
                # print(f"Mapped {key} -> {new_key}")
            else:
                print(f"Skipping layer tensor: {key}")
        
        # Handle Positional Embedding
        elif key == "positional_embedding.positional_embedding":
            new_key = "position_embd.weight"
            data = tensor.numpy().astype(np.float32)
            # Ensure shape is [MaxPos, D_Model]
            # Qwen uses generic sinusoids, typically max_source_positions
            if data.shape[0] < config.max_source_positions:
                 print(f"Warning: Positional embedding size {data.shape} smaller than max {config.max_source_positions}?")
            gguf_writer.add_tensor(new_key, data)
            added_count += 1
            print(f"Mapped Position Embedding -> {new_key}")
            
        # Handle Post Norm
        elif key == "ln_post.weight":
            new_key = "output_norm.weight" # or output_norm
            data = tensor.numpy().astype(np.float32)
            gguf_writer.add_tensor(new_key, data)
            added_count += 1
            print(f"Mapped {key} -> {new_key}")
        elif key == "ln_post.bias":
            new_key = "output_norm.bias"
            data = tensor.numpy().astype(np.float32)
            gguf_writer.add_tensor(new_key, data)
            added_count += 1
            
        # Handle Extra Heads (Projectors) - Saving as custom tensors
        elif key.startswith("proj") or key.startswith("conv"):
            # CNN layers (conv*) are skipped as per design (handled in Python)
            if key.startswith("conv"):
                # print(f"Skipping CNN tensor: {key}")
                continue
                
            # proj1 and proj2 are post-transformer projections
            # We save them so user can load them manually if needed
            new_key = f"extra.{key}"
            data = tensor.numpy().astype(np.float32)
            gguf_writer.add_tensor(new_key, data)
            added_count += 1
            print(f"Saved extra tensor: {key} -> {new_key}")
            
    # Add Tokenizer (Dummy) - BERT usually expects a tokenizer
    # We might need to add a dummy vocab to satisfy GGUF tools, 
    # even though we bypass it with embeddings.
    # Providing a minimal vocab.
    gguf_writer.add_tokenizer_model("bert") # or "no_vocab" if supported? 
    # Usually "llama" or "gpt2" tokenizer structure.
    # Let's add a dummy basic tokenizer
    vocab = ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"]
    gguf_writer.add_token_list(vocab)
    
    print(f"Total tensors exported: {added_count}")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    print("\n✅ Conversion complete!")
    print("""
    > [!NOTE]
    > The exported model uses 'bert' architecture.
    > The Qwen3-ASR Encoder is Pre-Norm, while Standard BERT is Post-Norm.
    > Please verify if inference matches expectations.
    > CNN tensors were skipped (conv2d1, conv2d2, conv2d3, conv_out).
    > Post-processing projections (proj1, proj2) were saved as 'extra.*' tensors.
    """)

if __name__ == "__main__":
    export_encoder_gguf()
