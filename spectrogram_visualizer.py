#!/usr/bin/env python3
"""
compare_spectrograms.py

Visual comparison of a linear-frequency STFT spectrogram and a log-Mel spectrogram
for a selected chunk from an audio file.

Usage examples:
  python compare_spectrograms.py --audio path/to/track.wav --offset 30 --duration 8
  python compare_spectrograms.py --audio path/to/track.mp3
  # If you don't have an audio file handy, generate a synthetic demo:
  python compare_spectrograms.py --synthetic --duration 6

Outputs:
  - normal_spectrogram.png
  - logmel_spectrogram.png
  - (optional) CSVs with the underlying matrices if --save-arrays is set
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Avoid seaborn entirely and do one chart per figure by design
import librosa
import librosa.display


def load_audio(args):
    if args.synthetic:
        sr = args.sr
        dur = args.duration
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # A simple “music-like” signal: two tones + a chirp + light noise
        y = (
            0.35 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 440 * t)
            + 0.20 * np.sin(2 * np.pi * (100 + 600 * t) * t)  # quadratic chirp-ish
        )
        y += 0.01 * np.random.randn(len(t))
        return y.astype(np.float32), sr
    else:
        if args.audio is None:
            raise ValueError("Please provide --audio PATH or use --synthetic.")
        # Load mono at target SR; offset/duration select the chunk
        y, sr = librosa.load(
            args.audio,
            sr=args.sr,
            mono=True,
            offset=max(args.offset, 0.0),
            duration=args.duration if args.duration > 0 else None,
        )
        return y, sr


def compute_linear_spectrogram(y, sr, n_fft, hop, fmin=20.0):
    # Power spectrogram (magnitude^2) on linear frequency bins
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def compute_logmel_spectrogram(y, sr, n_fft, hop, n_mels, fmin=20.0, fmax=None):
    M = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax,
    power=2.0,
    )
    M_db = librosa.power_to_db(M, ref=np.max)
    return M_db


def plot_and_save_spectrogram(S_db, sr, hop, title, fname, y_axis="log", fmin=20.0):
    plt.figure(figsize=(8, 4.2))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis=y_axis,  # "log" for STFT, "mel" for Mel
    )
    plt.title(title)
    plt.colorbar(format="%+2.0f dB", shrink=0.8)
    if y_axis == "log":
        # librosa handles log scaling internally; we just mention fmin in the title
        plt.ylabel("Frequency (log scale)")
    elif y_axis == "mel":
        plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (wav/mp3/flac).")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic demo signal.")
    parser.add_argument("--offset", type=float, default=0.0, help="Chunk start in seconds.")
    parser.add_argument("--duration", type=float, default=5.0, help="Chunk duration in seconds.")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate.")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT size.")
    parser.add_argument("--hop", type=int, default=512, help="Hop length.")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bands.")
    parser.add_argument("--fmin", type=float, default=20.0, help="Minimum frequency (Hz).")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency (Hz). Default=sr/2.")
    parser.add_argument("--save-arrays", action="store_true", help="Also save numpy arrays and CSVs.")
    args = parser.parse_args()

    y, sr = load_audio(args)
    fmax = args.fmax if args.fmax is not None else sr / 2.0

    # Compute spectrograms
    S_lin_db = compute_linear_spectrogram(y, sr, args.n_fft, args.hop, fmin=args.fmin)
    S_mel_db = compute_logmel_spectrogram(
        y, sr, args.n_fft, args.hop, args.n_mels, fmin=args.fmin, fmax=fmax
    )

    # Plot each in its own figure (no subplots, no seaborn, no explicit color selection)
    plot_and_save_spectrogram(
        S_lin_db, sr, args.hop,
        title=f"Linear-frequency STFT (n_fft={args.n_fft}, hop={args.hop})",
        fname="normal_spectrogram_16kHz.png",
        y_axis="log",  # log-frequency axis is standard for readability
        fmin=args.fmin,
    )

    plot_and_save_spectrogram(
        S_mel_db, sr, args.hop,
        title=f"Log-Mel Spectrogram (n_mels={args.n_mels})",
        fname="logmel_spectrogram_16kHz.png",
        y_axis="mel",
        fmin=args.fmin,
    )

    if args.save_arrays:
        # Save underlying matrices for reproducibility or inclusion in appendix
        np.save("normal_spectrogram_db.npy", S_lin_db)
        np.save("logmel_spectrogram_db.npy", S_mel_db)
        # Also CSVs (watch file size)
        np.savetxt("normal_spectrogram_db.csv", S_lin_db, delimiter=",")
        np.savetxt("logmel_spectrogram_db.csv", S_mel_db, delimiter=",")
        print("Saved numpy arrays and CSVs.")


if __name__ == "__main__":
    main()
