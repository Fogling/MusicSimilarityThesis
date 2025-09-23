#!/usr/bin/env python3
"""
Compute MIR feature summaries per subgenre (leaf folders) for a WAV dataset.

Features per track:
- duration_s
- spectral_centroid (mean, std)           [Hz]
- spectral_bandwidth (mean, std)          [Hz]
- zero_crossing_rate (mean, std)          [unitless, 0..1-ish]
- rms (mean, std)                         [amplitude]
- tempo_bpm (single global estimate)      [BPM]

Outputs:
- <out_dir>/<subgenre>_tracks.csv      # one row per track with all features
- <out_dir>/<subgenre>_summary.csv     # mean, std per feature + track_count
- <out_dir>/ALL_subgenres_summary.csv  # one row per subgenre (aggregated)

Usage:
    python compute_subgenre_mir_stats.py --wav_root WAV --out_dir stats
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

# ---------- Feature extraction ----------

def compute_track_features(path: Path,
                           sr_target: int | None = None,
                           n_fft: int = 2048,
                           hop_length: int = 512) -> dict:
    """
    Load audio and compute MIR features. Returns a dict with scalars.
    - sr_target=None keeps native sample rate (recommended here).
    """
    # Load mono for stable features
    y, sr = librosa.load(path.as_posix(), sr=sr_target, mono=True)
    # Trim leading/trailing near-silence (robustness to long tails)
    y, _ = librosa.effects.trim(y, top_db=30)
    duration_s = len(y) / sr if len(y) > 0 else 0.0

    if len(y) == 0:
        # Edge case: silent/empty file after trim
        return {
            "filename": path.name,
            "duration_s": 0.0,
            "centroid_mean_hz": np.nan, "centroid_std_hz": np.nan,
            "bandwidth_mean_hz": np.nan, "bandwidth_std_hz": np.nan,
            "zcr_mean": np.nan, "zcr_std": np.nan,
            "rms_mean": np.nan, "rms_std": np.nan,
            "tempo_bpm": np.nan,
        }

    # Frame-level features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)  # (1, T)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)  # (1, T)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)  # (1, T)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)  # (1, T)

    # Global tempo estimate (median is robust)
    try:
        tempo_arr = librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length, aggregate=None)
        tempo_bpm = float(np.median(tempo_arr)) if tempo_arr.size else np.nan
        # Optional refinement: prefer strong single-tempo by onset strength
        if np.isnan(tempo_bpm) or tempo_bpm <= 0:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempo_bpm = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length))
    except Exception:
        tempo_bpm = np.nan

    feat = {
        "filename": path.name,
        "duration_s": duration_s,
        "centroid_mean_hz": float(np.mean(centroid)),
        "centroid_std_hz": float(np.std(centroid, ddof=1)) if centroid.shape[1] > 1 else 0.0,
        "bandwidth_mean_hz": float(np.mean(bandwidth)),
        "bandwidth_std_hz": float(np.std(bandwidth, ddof=1)) if bandwidth.shape[1] > 1 else 0.0,
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr, ddof=1)) if zcr.shape[1] > 1 else 0.0,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms, ddof=1)) if rms.shape[1] > 1 else 0.0,
        "tempo_bpm": tempo_bpm,
    }
    return feat

# ---------- Directory walking ----------

def is_leaf_directory(dirpath: Path) -> bool:
    """A 'leaf' has at least one audio file and no subdirectories containing audio."""
    if any((dirpath / d).is_dir() for d in os.listdir(dirpath) if (dirpath / d).is_dir()):
        # has subfolders; we treat leaf as *deepest* level. We'll still allow
        # leaves even if subfolders exist, provided this folder has audio itself.
        pass
    # If there are audio files here and no audio files deeper, we’ll just treat every
    # deepest subdir as leaf in the main finder below. This helper is unused now.
    return True

def find_leaf_subgenre_dirs(root: Path) -> list[Path]:
    """
    Find deepest subdirectories that contain audio files.
    A leaf is defined as: contains at least one .wav and has no child directory
    that also contains .wav files.
    """
    audio_exts = {".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}
    all_dirs = [p for p in root.rglob("*") if p.is_dir()]
    # Keep only dirs that contain audio
    dirs_with_audio = [d for d in all_dirs if any((d / f).is_file() and (d / f).suffix.lower() in audio_exts
                                                  for f in os.listdir(d))]
    # Filter to deepest (no child in dirs_with_audio)
    leaf_dirs = []
    dirs_with_audio_set = set(map(str, dirs_with_audio))
    for d in dirs_with_audio:
        has_child_with_audio = any(str(child).startswith(str(d) + os.sep) and str(child) != str(d)
                                   for child in dirs_with_audio_set)
        if not has_child_with_audio:
            leaf_dirs.append(d)
    # If root itself contains audio and nothing deeper, include it
    if not leaf_dirs and any((root / f).suffix.lower() in audio_exts for f in os.listdir(root)):
        leaf_dirs = [root]
    return sorted(leaf_dirs)

# ---------- Aggregation & I/O ----------

def safe_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "-")

def summarize_subgenre(df: pd.DataFrame, subgenre: str) -> pd.DataFrame:
    agg = {}
    numeric_cols = [c for c in df.columns if c not in ("filename")]
    for col in numeric_cols:
        col_vals = df[col].astype(float)
        agg[f"{col}_mean"] = np.nanmean(col_vals)
        agg[f"{col}_std"] = np.nanstd(col_vals, ddof=1) if col_vals.shape[0] > 1 else 0.0
    agg["track_count"] = len(df)
    agg["subgenre"] = subgenre
    return pd.DataFrame([agg])

def main(wav_root: Path, out_dir: Path, sr_target: int | None, n_fft: int, hop_length: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    leaf_dirs = find_leaf_subgenre_dirs(wav_root)
    if not leaf_dirs:
        print(f"No subgenre folders with audio found under: {wav_root}")
        return

    all_summaries = []

    for subdir in leaf_dirs:
        subgenre = subdir.name
        audio_files = sorted([
            p for p in subdir.glob("*")
            if p.is_file() and p.suffix.lower() in {".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}
        ])
        if not audio_files:
            continue

        rows = []
        print(f"\nProcessing subgenre: {subgenre}  ({len(audio_files)} tracks)")
        for path in tqdm(audio_files, unit="track"):
            try:
                feat = compute_track_features(path, sr_target=sr_target, n_fft=n_fft, hop_length=hop_length)
                rows.append(feat)
            except Exception as e:
                print(f"[WARN] Failed on {path}: {e}")
                traceback.print_exc()

        if not rows:
            continue

        df_tracks = pd.DataFrame(rows)
        # Save per-subgenre tracks CSV
        subname = safe_name(subgenre)
        tracks_csv = out_dir / f"{subname}_tracks.csv"
        df_tracks.to_csv(tracks_csv, index=False)

        # Summary per subgenre
        df_summary = summarize_subgenre(df_tracks, subgenre=subgenre)
        summary_csv = out_dir / f"{subname}_summary.csv"
        df_summary.to_csv(summary_csv, index=False)

        all_summaries.append(df_summary)

    if all_summaries:
        df_all = pd.concat(all_summaries, ignore_index=True)
        df_all = df_all[["subgenre", "track_count"] + [c for c in df_all.columns if c not in {"subgenre", "track_count"}]]
        df_all.to_csv(out_dir / "ALL_subgenres_summary.csv", index=False)
        print(f"\nSaved master summary: {out_dir/'ALL_subgenres_summary.csv'}")
    else:
        print("No summaries produced (check your folder structure).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-subgenre MIR statistics from WAV dataset.")
    parser.add_argument("--wav_root", type=str, required=True, help="Path to WAV root (folder containing subgenre folders).")
    parser.add_argument("--out_dir", type=str, default="mir_stats", help="Where to write CSVs.")
    parser.add_argument("--sr", type=int, default=None, nargs="?", help="Target sample rate; default None keeps native.")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    args = parser.parse_args()

    main(Path(args.wav_root), Path(args.out_dir), args.sr, args.n_fft, args.hop_length)
