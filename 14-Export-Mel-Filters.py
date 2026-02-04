
import os
import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from export_config import MODEL_DIR, EXPORT_DIR

OUTPUT_FILE = EXPORT_DIR / "mel_filters.npy"

def main():
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    # The filters are usually in fe.mel_filters
    if hasattr(fe, 'mel_filters'):
        filters = np.array(fe.mel_filters)
        print(f"Filters shape: {filters.shape}")
        np.save(OUTPUT_FILE, filters)
        print(f"Saved filters to {OUTPUT_FILE}")
    else:
        # Fallback: calculate using the same logic as transformers
        from transformers.models.whisper.feature_extraction_whisper import mel_filter_bank
        # sr=16000, n_fft=400, n_mels=128
        filters = mel_filter_bank(
            num_frequency_bins=400 // 2 + 1,
            num_mel_filters=128,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
            mel_scale="slaney",
        )
        print(f"Calculated filters shape: {filters.shape}")
        np.save(OUTPUT_FILE, filters)
        print(f"Saved calculated filters to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
