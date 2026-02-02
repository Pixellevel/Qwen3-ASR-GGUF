
import os
import sys
import time
import torch
import librosa
import numpy as np
from threading import Thread
from pathlib import Path
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor, TextIteratorStreamer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import official model classes
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from qwen_asr.inference.utils import parse_asr_output

def main():
    # 1. Configuration
    HF_MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
    AUDIO_FILE = "test40.mp3"
    CHUNK_SIZE_SEC = 5.0
    
    # 官方流式参数
    UNFIXED_CHUNK_NUM = 2 # 前2个chunk不使用前缀
    UNFIXED_TOKEN_NUM = 5 # 每次回滚5个token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if not os.path.exists(HF_MODEL_DIR):
        print(f"Error: Model directory not found: {HF_MODEL_DIR}")
        return
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        return

    print(f"--- Loading Official Model (PyTorch {device}) ---")
    
    processor = AutoProcessor.from_pretrained(HF_MODEL_DIR, trust_remote_code=True, fix_mistral_regex=True)
    tokenizer = processor.tokenizer
    
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        HF_MODEL_DIR, 
        trust_remote_code=True, 
        torch_dtype=dtype
    ).to(device).eval()

    # 3. Load Audio
    print(f"Loading audio: {AUDIO_FILE}")
    full_audio, _ = librosa.load(AUDIO_FILE, sr=16000)
    
    # 4. Simulated Streaming Loop
    print(f"\n--- Starting Official-Style Streaming (Rollback & Text Streaming) ---")
    
    audio_accum = np.zeros((0,), dtype=np.float32)
    sample_rate = 16000
    chunk_samples = int(CHUNK_SIZE_SEC * sample_rate)
    
    total_chunks = int(np.ceil(len(full_audio) / chunk_samples))
    
    _raw_decoded = "" # 记录已解码文字
    
    for i in range(total_chunks):
        t_chunk_start = time.time()
        
        # 1. 获取当前音频块并累积
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(full_audio))
        chunk = full_audio[start:end]
        audio_accum = np.concatenate([audio_accum, chunk])
        
        # 2. 计算 Prefix (回滚逻辑)
        prefix = ""
        if i >= UNFIXED_CHUNK_NUM and _raw_decoded:
            actual_text = _raw_decoded.split("<asr_text>")[-1] if "<asr_text>" in _raw_decoded else _raw_decoded
            if actual_text:
                cur_ids = tokenizer.encode(actual_text)
                k = UNFIXED_TOKEN_NUM
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    prefix = tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    if '\ufffd' not in prefix:
                        break
                    if end_idx == 0:
                        prefix = ""
                        break
                    k += 1
        
        # 3. 构建 Prompt (使用官方模板方法)
        messages = [
            {"role": "system", "content": "语音转录："},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]}
        ]
        # apply_chat_template 会生成 <|im_start|>...<|im_start|>assistant\n
        base_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # 拼装最终 Prompt
        # 注意：官方逻辑中，<asr_text> 必须紧跟在 assistant 段落之后
        prompt = f"{base_prompt}<asr_text>{prefix}"
        
        # 4. 准备输入
        inputs = processor(text=prompt, audio=audio_accum, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if dtype == torch.float16:
            inputs["input_features"] = inputs["input_features"].to(dtype)

        # 5. 开启流式生成
        print(f"\n[{i+1}/{total_chunks}] {CHUNK_SIZE_SEC*(i+1):.1f}s | Prefix: '{prefix}'")
        print("Output: ", end="", flush=True)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        full_chunk_text = ""
        for new_text in streamer:
            # 过滤掉可能出现的重复 prefix (有些 streamer 行为可能不同)
            print(new_text, end="", flush=True)
            full_chunk_text += new_text
        
        # 更新状态：包含 prefix 的完整文本
        _raw_decoded = f"<asr_text>{prefix}{full_chunk_text}"
        print(f"\nCost: {time.time() - t_chunk_start:.2f}s")
        
    print("\n--- Streaming Done ---")

if __name__ == "__main__":
    main()
