#!/usr/bin/env python3
# stft_vs_logmel_comparison.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

def load_audio(path, sr, offset, duration, synthetic):
    if synthetic:
        t = np.linspace(0, duration, int(sr*duration), endpoint=False)
        # tones + chirp + noise to highlight differences
        y = (0.35*np.sin(2*np.pi*220*t)
             +0.25*np.sin(2*np.pi*440*t)
             +0.20*np.sin(2*np.pi*(100+600*t)*t))
        y += 0.01*np.random.randn(len(t))
        return y.astype(np.float32), sr
    y, sr = librosa.load(path, sr=sr, mono=True, offset=offset or 0.0,
                         duration=duration if duration>0 else None)
    return y, sr

def stft_db(y, sr, n_fft, hop):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
    return librosa.power_to_db(S, ref=np.max)

def mel_db(y, sr, n_fft, hop, n_mels, fmin=20.0, fmax=None):
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    return librosa.power_to_db(M, ref=np.max)

def mel_filterbank(sr, n_fft, n_mels, fmin=20.0, fmax=None):
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin,
                               fmax=fmax if fmax is not None else sr/2)

def set_cb(cb, label="Power (dB)"):
    cb.set_label(label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)
    args = ap.parse_args()

    y, sr = load_audio(args.audio, args.sr, args.offset, args.duration, args.synthetic)
    fmax = args.fmax if args.fmax is not None else sr/2

    S_db   = stft_db(y, sr, args.n_fft, args.hop)
    M_db   = mel_db(y, sr, args.n_fft, args.hop, args.n_mels, args.fmin, fmax)
    FB     = mel_filterbank(sr, args.n_fft, args.n_mels, args.fmin, fmax)

    # ---- Save individual figures (nice for thesis if you keep one-figure-per-plot) ----
    # (A) STFT linear Hz
    plt.figure(figsize=(9.5,4.2))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=args.hop, x_axis="time", y_axis="hz")
    cb = plt.colorbar(img, format="%+2.0f dB", shrink=0.85); set_cb(cb)
    ax = plt.gca()
    ax.set_yticklabels([f"{t/1000:.1f}" for t in ax.get_yticks()])
    plt.ylabel("Frequency (kHz)"); plt.xlabel("Time (s)")
    plt.title(f"STFT spectrogram (linear frequency)  n_fft={args.n_fft}, hop={args.hop}")
    plt.tight_layout(); plt.savefig("stft_linear_hz.png", dpi=220); plt.close()

    # (B) STFT log-frequency axis
    plt.figure(figsize=(9.5,4.2))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=args.hop, x_axis="time", y_axis="log")
    cb = plt.colorbar(img, format="%+2.0f dB", shrink=0.85); set_cb(cb)
    plt.ylabel("Frequency (Hz, log scale)"); plt.xlabel("Time (s)")
    plt.title("STFT spectrogram (log-scaled axis)")
    plt.tight_layout(); plt.savefig("stft_log_axis.png", dpi=220); plt.close()

    # (C) Log-Mel spectrogram
    plt.figure(figsize=(9.5,4.2))
    img = librosa.display.specshow(M_db, sr=sr, hop_length=args.hop, x_axis="time", y_axis="mel")
    cb = plt.colorbar(img, format="%+2.0f dB", shrink=0.85); set_cb(cb)
    plt.ylabel("Frequency (Mel)"); plt.xlabel("Time (s)")
    plt.title(f"Log-Mel spectrogram (n_mels={args.n_mels})")
    plt.tight_layout(); plt.savefig("logmel.png", dpi=220); plt.close()

    # (D) Mel filterbank heatmap
    plt.figure(figsize=(8.5,3.8))
    plt.imshow(FB, aspect="auto", origin="lower",
               extent=[0, sr/2, 0, args.n_mels])  # x axis in Hz
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Mel band index")
    plt.title(f"Mel filterbank (n_mels={args.n_mels})")
    plt.tight_layout(); plt.savefig("mel_filterbank.png", dpi=220); plt.close()

    # ---- 3-panel comparison figure (for side-by-side in one page) ----
    fig, axs = plt.subplots(1, 3, figsize=(16,4.6), constrained_layout=True)
    # A
    img0 = librosa.display.specshow(S_db, sr=sr, hop_length=args.hop, x_axis="time", y_axis="hz", ax=axs[0])
    axs[0].set_title("A) STFT (linear frequency)")
    axs[0].set_xlabel("Time (s)")
    yt = axs[0].get_yticks(); axs[0].set_yticklabels([f"{t/1000:.1f}" for t in yt])
    axs[0].set_ylabel("Frequency (kHz)")
    # B
    img1 = librosa.display.specshow(S_db, sr=sr, hop_length=args.hop, x_axis="time", y_axis="log", ax=axs[1])
    axs[1].set_title("B) STFT (log-scaled axis)")
    axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Frequency (Hz, log)")
    # C
    img2 = librosa.display.specshow(M_db, sr=sr, hop_length=args.hop, x_axis="time", y_axis="mel", ax=axs[2])
    axs[2].set_title(f"C) Log-Mel (n_mels={args.n_mels})")
    axs[2].set_xlabel("Time (s)"); axs[2].set_ylabel("Frequency (Mel)")
    # single shared colorbar on the right
    cbar = fig.colorbar(img2, ax=axs.ravel().tolist(), format="%+2.0f dB", shrink=0.9, pad=0.02)
    cbar.set_label("Power (dB)")
    fig.savefig("stft_vs_logmel_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
