import os
import re
import torch
import torchaudio
from tqdm import tqdm
from transformers import ASTFeatureExtractor

def sanitize_filename(filename):
    sanitized = re.sub(r'[\\/:*?"<>|]', '', filename)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized[:100]

def preprocess_and_extract_features(
    root_dir,
    output_dir="preprocessed_features",
    chunk_duration=15,
    chunk_gap=10,
    target_sr=16000,
    extractor_name="MIT/ast-finetuned-audioset-10-10-0.4593"
):
    os.makedirs(output_dir, exist_ok=True)
    feature_extractor = ASTFeatureExtractor.from_pretrained(extractor_name)

    for genre_folder in os.listdir(root_dir):
        genre_path = os.path.join(root_dir, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        for subgenre_folder in os.listdir(genre_path):
            subgenre_path = os.path.join(genre_path, subgenre_folder)
            if not os.path.isdir(subgenre_path):
                continue

            label_dir = os.path.join(output_dir, subgenre_folder)
            os.makedirs(label_dir, exist_ok=True)

            files = [f for f in os.listdir(subgenre_path) if f.lower().endswith((".mp3", ".wav"))]
            for filename in tqdm(files, desc=f"Extracting {subgenre_folder}"):
                file_path = os.path.join(subgenre_path, filename)
                try:
                    waveform, sr = torchaudio.load(file_path)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                        waveform = resampler(waveform)

                    waveform = waveform.squeeze()
                    chunk_samples = chunk_duration * target_sr
                    for i in range(3):
                        start_sample = i * (chunk_duration + chunk_gap) * target_sr
                        end_sample = start_sample + chunk_samples
                        if end_sample > waveform.size(0):
                            break

                        chunk = waveform[start_sample:end_sample].numpy()
                        inputs = feature_extractor(chunk, sampling_rate=target_sr, return_tensors="pt", padding="max_length")

                        base_name = sanitize_filename(os.path.splitext(filename)[0])
                        out_path = os.path.join(label_dir, f"{base_name}_chunk{i+1}.pt")
                        torch.save(inputs, out_path)

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    preprocess_and_extract_features("MP3")
