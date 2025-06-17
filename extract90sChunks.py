import torchaudio
import torch
import torchaudio.transforms as T

def extract_90s_chunks(audio_path, target_sr=16000, chunk_count=3):
    waveform, sr = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    total_samples = waveform.size(0)
    target_duration = 90  # seconds
    target_samples = target_duration * target_sr

    if total_samples < target_samples:
        # Pad to 90s if too short
        padding = torch.zeros(target_samples - total_samples)
        waveform = torch.cat([waveform, padding])
        start_idx = 0
    else:
        # Take middle 90 seconds
        start_idx = (total_samples - target_samples) // 2

    window = waveform[start_idx: start_idx + target_samples]

    # Split into 3 equal chunks
    chunk_samples = target_samples // chunk_count
    chunks = [window[i * chunk_samples : (i + 1) * chunk_samples] for i in range(chunk_count)]

    return chunks
