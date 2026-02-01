import gguf
import sys

def dump_gguf(path):
    reader = gguf.GGUFReader(path)
    print(f"--- Metadata for {path} ---")
    for key in reader.fields:
        field = reader.fields[key]
        # Skip tensors to avoid clutter
        if key.startswith("tensor"): continue
        
        val = field.parts[-1]
        print(f"{key}: {val}")

if __name__ == "__main__":
    dump_gguf("model/qwen3_asr_llm.f16.gguf")
