import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T

torchaudio.set_audio_backend("soundfile")

# Optional subgenre-specific start times (extend as needed)
start_seconds = {
    "Chill House": 30,
    "Banger House": 30,
    "Party Techno": 0,
    "Schiebender Techno": 60,
    "Emotional + melancholic Techno": 0,
    "Hard Techno": 10,
    "Banger Goa": 15,
    "Chiller vibe Goa": 30,
    "Zyzz Music": 0,
}

def chunk_folder(subgenre_path, target_sr=16000, window_duration=75, chunk_count=3, output_dir="preprocessed_chunks"):
    if not os.path.isdir(subgenre_path):
        print("Error: Not a valid folder:", subgenre_path)
        return

    subgenre = os.path.basename(subgenre_path)
    output_path = os.path.join(output_dir, subgenre)
    os.makedirs(output_path, exist_ok=True)

    total_samples = window_duration * target_sr
    chunk_len = total_samples // chunk_count
    start_sec = start_seconds.get(subgenre, None)

    for filename in os.listdir(subgenre_path):
        if not filename.lower().endswith(".mp3"):
            continue
        file_path = os.path.join(subgenre_path, filename)
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                waveform = resampler(waveform)

            if waveform.size(-1) < total_samples:
                pad = torch.zeros(total_samples - waveform.size(-1))
                waveform = torch.cat([waveform, pad])
                start_sample = 0
            else:
                start_sample = (
                    min(start_sec * target_sr, waveform.size(-1) - total_samples)
                    if start_sec is not None
                    else (waveform.size(-1) - total_samples) // 2
                )

            window = waveform[start_sample: start_sample + total_samples]

            for i in range(chunk_count):
                chunk = window[i * chunk_len: (i + 1) * chunk_len]
                out_name = f"{os.path.splitext(filename)[0]}_chunk{i+1}.pt"
                torch.save(chunk, os.path.join(output_path, out_name))

            print(f"✓ {filename}")

        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunk_folder.py <path_to_subgenre_folder>")
        sys.exit(1)
    chunk_folder(sys.argv[1])
