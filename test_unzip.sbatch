#!/bin/bash
#SBATCH --partition=study
#SBATCH --job-name=ast-finetune
#SBATCH --gres=gpu:gtx:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --tmp=100G
#SBATCH --time=1-00:00:00

#exit on error or undef variables
set -euo pipefail

######### EDIT THESE #########
ARCHIVE="$HOME/thesis/precomputed_AST.zip"
CONFIG_JSON="$HOME/thesis/train_from_precomputed.json"
################################

######### EDIT THESE #########
ARCHIVE="$HOME/thesis/precomputed_AST.zip"       # supports .zip, .tar.gz, .tgz
################################

echo "=== Starting staging test ==="
echo "[INFO] SLURM_JOB_TMP = ${SLURM_JOB_TMP}"

# 1) Check source archive exists
if [[ ! -f "$ARCHIVE" ]]; then
    echo "[ERROR] Archive not found: $ARCHIVE"
    exit 1
fi
echo "[OK] Found archive at $ARCHIVE"

# 2) Copy archive into node-local SSD
cp -v "$ARCHIVE" "$SLURM_JOB_TMP/"
echo "[OK] Copied archive to $SLURM_JOB_TMP"

# 3) Unpack into SSD
cd "$SLURM_JOB_TMP"
fname="$(basename "$ARCHIVE")"
if [[ "$fname" == *.zip ]]; then
    unzip -q "$fname"
elif [[ "$fname" == *.tar.gz ]] || [[ "$fname" == *.tgz ]]; then
    tar -xzf "$fname"
else
    echo "[ERROR] Unsupported archive type: $fname"
    exit 2
fi
echo "[OK] Unpacked archive"

# 4) Show extracted structure (top 2 levels)
echo "=== Extracted tree ==="
find "$SLURM_JOB_TMP" -maxdepth 2 -type d -print
find "$SLURM_JOB_TMP" -maxdepth 2 -type f | head -n 10

echo "=== Done ==="