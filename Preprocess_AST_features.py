import os
import re
import json
from typing import List, Tuple

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import ASTFeatureExtractor

def sanitize_filename(filename):
    sanitized = re.sub(r'[\\/:*?"<>|]', '', filename)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized[:100]

def list_audio_files(root: str, exts=(".mp3", ".wav")) -> List[Tuple[str, str, str]]:
    """
    Returns list of (subgenre, filepath, basename) for all audio files
    under <root>/<genre>/<subgenre>/*.{exts}
    """
    items = []
    for genre in os.listdir(root):
        gpath = os.path.join(root, genre)
        if not os.path.isdir(gpath):
            continue
        for sub in os.listdir(gpath):
            spath = os.path.join(gpath, sub)
            if not os.path.isdir(spath):
                continue
            for f in os.listdir(spath):
                if f.lower().endswith(exts):
                    items.append((sub, os.path.join(spath, f), os.path.splitext(f)[0]))
    return items

def load_and_resample(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav.squeeze(0)

def chunk_waveform(waveform: torch.Tensor, sr: int, duration_s: int, max_chunks: int = 9):
    """
    Yield up to max_chunks non-overlapping chunks of duration duration_s seconds.
    """
    size = duration_s * sr
    t = waveform.shape[0]
    for i in range(max_chunks):
        start = i * size
        end = start + size
        if end > t:
            break
        yield waveform[start:end]

def compute_stats_pass(root_dir: str, target_sr: int, chunk_duration: int) -> Tuple[float, float]:
    """
    Pass 1: compute mean/std over *pre-normalized* AST inputs ("input_values")
    by running the extractor with do_normalize=False.
    """
    print("Pass 1/2: Computing dataset mean/std (do_normalize=False)...")
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    if hasattr(extractor, "do_normalize"):
        extractor.do_normalize = False

    means, stds = [], []
    items = list_audio_files(root_dir)
    for sub, fpath, base in tqdm(items, desc="Stats pass"):
        try:
            wav = load_and_resample(fpath, target_sr)
            for chunk in chunk_waveform(wav, target_sr, chunk_duration):
                inputs = extractor(
                    chunk.numpy(),
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding="max_length"
                )
                x = inputs["input_values"][0].float()
                means.append(x.mean().item())
                stds.append(x.std(unbiased=False).item())
        except Exception as e:
            print(f"[stats] Skipping {fpath}: {e}")

    mean_val = float(np.mean(means)) if means else 0.0
    std_val = float(np.mean(stds)) if stds else 1.0
    print(f"Dataset stats — mean: {mean_val:.6f}, std: {std_val:.6f}")
    return mean_val, std_val

def preprocess_and_extract_features(
    root_dir,
    output_dir="preprocessed_features",
    chunk_duration=10,
    target_sr=16000,
    extractor_name="MIT/ast-finetuned-audioset-10-10-0.4593"
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- PASS 1: compute dataset stats on pre-normalized features
    mean_val, std_val = compute_stats_pass(
        root_dir=root_dir,
        target_sr=target_sr,
        chunk_duration=chunk_duration
    )

    # ---- Create and configure extractor with dataset stats for PASS 2
    feature_extractor = ASTFeatureExtractor.from_pretrained(extractor_name)
    feature_extractor.mean = mean_val
    feature_extractor.std = std_val
    if hasattr(feature_extractor, "do_normalize"):
        feature_extractor.do_normalize = True

    # Save stats for reference/reuse
    with open(os.path.join(output_dir, "feature_stats.json"), "w") as f:
        json.dump({"mean": mean_val, "std": std_val, "sr": target_sr}, f, indent=2)

    # ---- PASS 2: extract & save normalized features once
    print("Pass 2/2: Extracting and saving normalized features...")
    items = list_audio_files(root_dir)
    for sub, fpath, base in tqdm(items, desc="Extracting"):
        try:
            label_dir = os.path.join(output_dir, sub)
            os.makedirs(label_dir, exist_ok=True)

            wav = load_and_resample(fpath, target_sr)
            chunk_idx = 1
            for chunk in chunk_waveform(wav, target_sr, chunk_duration):
                inputs = feature_extractor(
                    chunk.numpy(),
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding="max_length"
                )
                out_name = f"{sanitize_filename(base)}_chunk{chunk_idx}.pt"
                out_path = os.path.join(label_dir, out_name)
                torch.save(dict(inputs), out_path)
                chunk_idx += 1

        except Exception as e:
            print(f"[extract] Failed {fpath}: {e}")

if __name__ == "__main__":
    preprocess_and_extract_features(root_dir="WAV")
