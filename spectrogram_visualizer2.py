#!/usr/bin/env python3
# stft_log_vs_logmel_2panel.py
"""
Two-panel comparison:
  A) STFT spectrogram (log-scaled frequency axis, Hz)
  B) Log-Mel spectrogram (n_mels=128)

Designed to be compact in theses: narrow margins, shared colorbar.

Usage
  python stft_log_vs_logmel_2panel.py --audio path/to/track.wav --offset 30 --duration 8
  # or a didactic synthetic signal:
  python stft_log_vs_logmel_2panel.py --synthetic --duration 6

Outputs
  - stft_log_vs_logmel_2panel.png
  - (optional) individual: stft_log.png, logmel.png  (use --save-individual)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

def load_audio(args):
    if args.synthetic:
        sr = args.sr
        t = np.linspace(0, args.duration, int(sr*args.duration), endpoint=False)
        # tones + rising chirp + light noise (nice for showing differences)
        y = (0.35*np.sin(2*np.pi*220*t)
             +0.25*np.sin(2*np.pi*440*t)
             +0.20*np.sin(2*np.pi*(100+600*t)*t))
        y += 0.01*np.random.randn(len(t))
        return y.astype(np.float32), sr
    if not args.audio:
        raise ValueError("Provide --audio PATH or use --synthetic.")
    y, sr = librosa.load(args.audio, sr=args.sr, mono=True,
                         offset=max(args.offset,0.0),
                         duration=args.duration if args.duration>0 else None)
    return y, sr

def stft_db(y, n_fft, hop):
    P = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
    return librosa.power_to_db(P, ref=np.max)

def mel_db(y, sr, n_fft, hop, n_mels, fmin, fmax):
    M = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    return librosa.power_to_db(M, ref=np.max)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sr", type=int, default=16000)      # AST-friendly
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--n_fft", type=int, default=4096)    # sharp visuals
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)   # default sr/2
    ap.add_argument("--dpi", type=int, default=300)       # <- default high quality
    ap.add_argument("--save-individual", action="store_true")
    args = ap.parse_args()

    y, sr = load_audio(args)
    fmax = args.fmax if args.fmax is not None else sr/2

    S_db  = stft_db(y, args.n_fft, args.hop)
    M_db  = mel_db(y, sr, args.n_fft, args.hop, args.n_mels, args.fmin, fmax)

    # ---------- INDIVIDUAL (optional) ----------
    if args.save_individual:
        # STFT (log axis)
        plt.figure(figsize=(6.2,3.6))
        im = librosa.display.specshow(S_db, sr=sr, hop_length=args.hop,
                                      x_axis="time", y_axis="log")
        cb = plt.colorbar(im, format="%+2.0f dB", shrink=0.85)
        cb.set_label("Power (dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz, log)")
        plt.title("STFT (log-scaled axis)")
        plt.tight_layout()
        plt.savefig("stft_log.png", dpi=args.dpi); plt.close()

        # Log-Mel
        plt.figure(figsize=(6.2,3.6))
        im = librosa.display.specshow(M_db, sr=sr, hop_length=args.hop,
                                      x_axis="time", y_axis="mel")
        cb = plt.colorbar(im, format="%+2.0f dB", shrink=0.85)
        cb.set_label("Power (dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Mel)")
        plt.title(f"Log-Mel (n_mels={args.n_mels})")
        plt.tight_layout()
        plt.savefig("logmel.png", dpi=args.dpi); plt.close()

    # ---------- COMPACT 2-PANEL FIGURE ----------
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)

    # A) STFT (log axis, Hz)
    im0 = librosa.display.specshow(S_db, sr=sr, hop_length=args.hop,
                                   x_axis="time", y_axis="log", ax=axs[0])
    axs[0].set_title("A) STFT (log-scaled axis)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz, log)")

    # B) Log-Mel (Mel units)
    im1 = librosa.display.specshow(M_db, sr=sr, hop_length=args.hop,
                                   x_axis="time", y_axis="mel", ax=axs[1])
    axs[1].set_title(f"B) Log-Mel (n_mels={args.n_mels})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Mel)")

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), format="%+2.0f dB",
                        shrink=0.95, pad=0.02)
    cbar.set_label("Power (dB)")

    # Save at thesis-quality DPI
    fig.savefig("stft_log_vs_logmel_2panel.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved stft_log_vs_logmel_2panel.png")

if __name__ == "__main__":
    main()
