#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def to_float_nan(v):
    if v is None:
        return np.nan
    if isinstance(v, str) and v.strip().lower() in {"", "null", "nan"}:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def infer_experiment_name_from_filename(p: Path) -> str:
    """Try to infer a human-readable experiment name from filename."""
    name = p.stem
    m = re.search(r'(\b|_)([234])\s*chunks?(\b|_)', name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(2)} chunks"
    m = re.search(r'([234])\s*chunk', name, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} chunks"
    return re.sub(r'[_\-]+', ' ', name).strip()

def collect_final_stats(in_dir: Path):
    """Find CSVs that look like final/summary stats and return list of dicts with name, mean, std."""
    stats = []
    patterns = ["*final*stats*.csv", "*summary*.csv", "*overall*.csv", "*fold*_summary*.csv"]
    files = []
    for pat in patterns:
        files.extend(in_dir.glob(pat))
    files = sorted(set(files))
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        cand_mean = next((cols[c] for c in cols if "mean" in c and "acc" in c), None)
        cand_std  = next((cols[c] for c in cols if ("std" in c or "stdev" in c) and "acc" in c), None)
        if cand_mean is None and "accuracy" in cols:
            cand_mean = cols["accuracy"]
        name_col = next((cols[c] for c in cols if "experiment" in c or "name" in c), None)
        for _, row in df.iterrows():
            mean_val = to_float_nan(row[cand_mean]) if cand_mean else np.nan
            std_val  = to_float_nan(row[cand_std]) if cand_std else np.nan
            if np.isnan(mean_val):
                continue
            if name_col:
                name = str(row[name_col])
            else:
                name = infer_experiment_name_from_filename(p)
            stats.append({"name": name, "mean": mean_val, "std": std_val})
    return stats

def group_histories(in_dir: Path):
    """
    Collect per-epoch histories from CSVs (and optionally JSONs) and group by experiment name.
    Returns dict: {exp_name: list of DataFrames}, where each DF is a fold with columns epoch, eval_accuracy, train_loss.
    """
    groups = {}

    def add_history(name, df):
        c = {col.lower(): col for col in df.columns}
        epoch_col = c.get("epoch") or c.get("epochs") or c.get("step") or c.get("steps")
        eval_col  = c.get("eval_accuracy") or c.get("eval_acc") or c.get("val_accuracy") or c.get("val_acc")
        loss_col  = c.get("train_loss") or c.get("loss") or c.get("training_loss")
        if epoch_col is None:
            df["_epoch"] = np.arange(len(df))
            epoch_col = "_epoch"
        df_use = pd.DataFrame({
            "epoch": coerce_numeric(df[epoch_col]),
            "eval_accuracy": coerce_numeric(df[eval_col]) if eval_col in df else pd.Series([np.nan]*len(df)),
            "train_loss": coerce_numeric(df[loss_col]) if loss_col in df else pd.Series([np.nan]*len(df)),
        })
        df_use = df_use.sort_values("epoch").reset_index(drop=True)
        groups.setdefault(name, []).append(df_use)

    for p in sorted(in_dir.glob("*.csv")):
        if re.search(r'(final|summary|overall)', p.name, flags=re.IGNORECASE):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        name = infer_experiment_name_from_filename(p)
        add_history(name, df)

    for p in sorted(in_dir.glob("*.json")):
        try:
            obj = json.load(open(p, "r"))
        except Exception:
            continue
        name = obj.get("experiment_name") or infer_experiment_name_from_filename(p)
        folds = obj.get("fold_metrics", [])
        if not folds:
            continue
        for fm in folds:
            hist = fm.get("training_history", {})
            epochs = pd.Series([to_float_nan(v) for v in hist.get("epoch", [])], name="epoch")
            evala  = pd.Series([to_float_nan(v) for v in hist.get("eval_accuracy", [])], name="eval_accuracy")
            tloss  = pd.Series([to_float_nan(v) for v in hist.get("train_loss", [])], name="train_loss")
            df = pd.concat([epochs, evala, tloss], axis=1)
            add_history(name, df)

    return groups

def aggregate_histories(dfs):
    """Given list of fold DataFrames (epoch, eval_accuracy, train_loss), return epochs, mean/std for each metric (NaN-safe)."""
    common_epochs = None
    clean_dfs = []
    for df in dfs:
        dfc = df.dropna(subset=["epoch"]).copy()
        dfc["epoch"] = dfc["epoch"].astype(float)
        clean_dfs.append(dfc)
        e = dfc["epoch"].values
        if common_epochs is None:
            common_epochs = e
        else:
            if len(e) != len(common_epochs) or np.any(e != common_epochs):
                common_epochs = None
    if common_epochs is None:
        minlen = min(len(df) for df in clean_dfs)
        arr_eval = np.vstack([df["eval_accuracy"].values[:minlen] for df in clean_dfs]).astype(float)
        arr_loss = np.vstack([df["train_loss"].values[:minlen] for df in clean_dfs]).astype(float)
        epochs   = np.arange(minlen, dtype=float)
    else:
        minlen = min(len(df) for df in clean_dfs)
        common = set(clean_dfs[0]["epoch"].values)
        for df in clean_dfs[1:]:
            common &= set(df["epoch"].values)
        if not common:
            minlen = min(len(df) for df in clean_dfs)
            arr_eval = np.vstack([df["eval_accuracy"].values[:minlen] for df in clean_dfs]).astype(float)
            arr_loss = np.vstack([df["train_loss"].values[:minlen] for df in clean_dfs]).astype(float)
            epochs   = np.arange(minlen, dtype=float)
        else:
            common = np.array(sorted(common), dtype=float)
            eval_list, loss_list = [], []
            for df in clean_dfs:
                dfc = df.set_index("epoch").reindex(common)
                eval_list.append(dfc["eval_accuracy"].values)
                loss_list.append(dfc["train_loss"].values)
            arr_eval = np.vstack(eval_list).astype(float)
            arr_loss = np.vstack(loss_list).astype(float)
            epochs   = common

    eval_mean = np.nanmean(arr_eval, axis=0)
    eval_std  = np.nanstd(arr_eval, axis=0)
    loss_mean = np.nanmean(arr_loss, axis=0)
    loss_std  = np.nanstd(arr_loss, axis=0)
    return epochs, eval_mean, eval_std, loss_mean, loss_std

def plot_bar(stats, out_dir: Path, title="Chunking Setup: Mean Accuracy ± STD"):
    def chunk_num(name):
        m = re.search(r'(\d+)\s*chunks?', name, flags=re.IGNORECASE)
        return int(m.group(1)) if m else None

    with_nums = [(s, chunk_num(s["name"])) for s in stats]
    if all(n is not None for _, n in with_nums):
        with_nums.sort(key=lambda x: x[1])
        stats_sorted = [s for s, _ in with_nums]
    else:
        stats_sorted = sorted(stats, key=lambda s: s["mean"])

    labels = [f"{chunk_num(s['name'])} Chunks" if chunk_num(s["name"]) else s["name"]
              for s in stats_sorted]
    means  = [s["mean"] for s in stats_sorted]
    stds   = [s.get("std", np.nan) for s in stats_sorted]

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    # width < 1.0 makes bars thinner
    plt.bar(x, means, yerr=stds, width=0.35, align='center', capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Mean Accuracy (5-fold)")
    plt.title(title)
    plt.grid(axis="y")
    out_path = out_dir / "chunking_mean_accuracy_bar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def plot_histories(groups, out_dir: Path):
    out_paths = []
    for name, dfs in groups.items():
        if not dfs:
            continue
        epochs, eval_mean, eval_std, loss_mean, loss_std = aggregate_histories(dfs)

        fig = plt.figure(figsize=(7, 3.8))
        ax1 = plt.gca()

        if epochs is not None and np.isfinite(eval_mean).any():
            ax1.plot(epochs, eval_mean, label="Eval Accuracy")
            if np.isfinite(eval_std).any():
                ax1.fill_between(epochs, eval_mean - eval_std, eval_mean + eval_std, alpha=0.2)
            ax1.set_ylabel("Accuracy")
            ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        if epochs is not None and np.isfinite(loss_mean).any():
            ax2.plot(epochs, loss_mean, linestyle="--", label="Train Loss")
            if np.isfinite(loss_std).any():
                ax2.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.15)
            ax2.set_ylabel("Train Loss")

        ax1.set_xlabel("Epoch")
        ax1.set_title(f"{name}: Eval Accuracy & Train Loss")
        ax1.grid(True)

        lines, labs = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2
        labs += lb2
        if lines:
            ax1.legend(lines, labs, loc="best")

        out_name = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_') + "_acc_loss.png"
        out_path = out_dir / out_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        out_paths.append(out_path)
    return out_paths

def load_json_fallback(in_dir: Path):
    stats = []
    for p in sorted(in_dir.glob("*.json")):
        try:
            obj = json.load(open(p, "r"))
        except Exception:
            continue
        name = obj.get("experiment_name") or infer_experiment_name_from_filename(p)
        fs = obj.get("final_statistics", {})
        mean = to_float_nan(fs.get("mean_accuracy", np.nan))
        std  = to_float_nan(fs.get("std_accuracy", np.nan))
        if not np.isnan(mean):
            stats.append({"name": name, "mean": mean, "std": std})
    return stats

def main():
    ap = argparse.ArgumentParser(description="Plot chunking experiment results from CSV summaries (NaN-safe).")
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing CSV (and/or JSON) summaries.")
    ap.add_argument("--out_dir", type=str, default=None, help="Directory to save plots (default: in_dir).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = collect_final_stats(in_dir)
    if not stats:
        stats = load_json_fallback(in_dir)

    bar_path = None
    if stats:
        bar_path = plot_bar(stats, out_dir)

    groups = group_histories(in_dir)
    history_paths = plot_histories(groups, out_dir) if groups else []

    print("[OK] Plots written to:", out_dir)
    if bar_path:
        print(" -", bar_path)
    for p in history_paths:
        print(" -", p)

if __name__ == "__main__":
    main()
