import os
import torchaudio
import torch
import torchaudio.transforms as T

# Optional: Define custom start seconds for each subgenre
start_seconds = {
    "Chill House": 30,
    "Banger House": 30,
    "Party Techno": 0,
    "Schiebender Techno": 60,
    "Emotional + melancholic Techno": 0,
    "Hard Techno": 10,
    "Banger Goa": 30,
    "Chiller vibe Goa": 30,
    "Arsch Goa": 0,
    # Add more entries as needed
}

def preprocess_music_folder(root_dir, target_sr=16000, window_duration=75, chunk_count=3, output_dir="preprocessed_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    chunk_duration = window_duration // chunk_count
    chunk_samples = chunk_duration * target_sr
    total_samples = window_duration * target_sr

    for genre_folder in os.listdir(root_dir):
        genre_path = os.path.join(root_dir, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        for subgenre_folder in os.listdir(genre_path):
            subgenre_path = os.path.join(genre_path, subgenre_folder)
            if not os.path.isdir(subgenre_path):
                continue

            label = subgenre_folder
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            for filename in os.listdir(subgenre_path):
                if not filename.lower().endswith(".mp3"):
                    continue

                file_path = os.path.join(subgenre_path, filename)
                try:
                    waveform, sr = torchaudio.load(file_path)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(dim=0)

                    if sr != target_sr:
                        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                        waveform = resampler(waveform)

                    total_audio_samples = waveform.size(0)

                    if total_audio_samples < total_samples:
                        padding = torch.zeros(total_samples - total_audio_samples)
                        waveform = torch.cat([waveform, padding])
                        start_sample = 0
                    else:
                        start_sec = start_seconds.get(subgenre_folder, None)
                        if start_sec is not None:
                            start_sample = min(start_sec * target_sr, total_audio_samples - total_samples)
                        else:
                            start_sample = (total_audio_samples - total_samples) // 2

                    window = waveform[start_sample: start_sample + total_samples]
                    chunk_len = total_samples // chunk_count

                    for i in range(chunk_count):
                        chunk = window[i * chunk_len : (i + 1) * chunk_len]
                        out_path = os.path.join(label_dir, f"{os.path.splitext(filename)[0]}_chunk{i+1}.pt")
                        torch.save(chunk, out_path)

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    preprocess_music_folder("MP3")
