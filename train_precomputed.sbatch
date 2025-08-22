#!/bin/bash
#SBATCH --partition=study
#SBATCH --job-name=ast-finetune
#SBATCH -o %x.%j.log
#SBATCH -e %o
#SBATCH --gres=gpu:gtx:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --tmp=100G
#SBATCH --time=1-00:00:00

#exit on error or undef variables
set -euo pipefail

######### EDIT THESE #########
# 1) Where the archive lives on the cluster (shared disk)
ARCHIVE="~/thesis/precomputed_AST.zip"        # or .tar.gz
# 2) A shared directory to collect results after the job
OUT_BASE="/path/to/ast_runs"
# 3) Your training script and config (already in repo/workdir)
TRAIN_SCRIPT="AST_Triplet_training.py"
CONFIG_JSON="train_from_precomputed.json"     # will be copied & patched
################################

# Guard against thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Scratch on node-local SSD
SCRATCH="${SLURM_JOB_TMP:-/tmp}/ast_job_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"/{archive,features,run,logs}
echo "[INFO] SCRATCH: $SCRATCH"

echo "[STEP] Copy archive to SSD..."
cp -f "$ARCHIVE" "$SCRATCH/archive/"

echo "[STEP] Unpack archive..."
cd "$SCRATCH/preprocessed_features"
fname="$(basename "$ARCHIVE")"
if [[ "$fname" == *.zip ]]; then
    unzip -q "../archive/$fname"
elif [[ "$fname" == *.tar.gz ]] || [[ "$fname" == *.tgz ]]; then
    tar -xzf "../archive/$fname"
else
    echo "[ERROR] Unsupported archive type: $fname"
    exit 2
fi

# Detect the top-level extracted folder (assumes you zipped a folder, not just loose files)
# If you followed the advice, this should be 'precomputed_AST'
TOPDIR="$(find . -maxdepth 1 -mindepth 1 -type d | head -n1)"
if [[ -z "$TOPDIR" ]]; then
    echo "[ERROR] Could not find extracted feature folder."
    exit 3
fi

# Sanity check: we should see .pt files somewhere beneath TOPDIR
PT_COUNT="$(find "$TOPDIR" -type f -name "*.pt" | wc -l)"
if [[ "$PT_COUNT" -lt 1 ]]; then
    echo "[ERROR] No .pt files found under $TOPDIR"
    find "$TOPDIR" -maxdepth 2 -type d -print
    exit 4
fi
FEATURES_DIR="$(realpath "$TOPDIR")"
echo "[INFO] Features dir: $FEATURES_DIR  (pt files: $PT_COUNT)"

# Prepare a job-local config that points to the SSD copy
echo "[STEP] Prepare config..."
cp -f "$CONFIG_JSON" "$SCRATCH/run/config.json"
# Patch data.chunks_dir in JSON (requires jq; most clusters have it. If not, see note below)
jq --arg p "$FEATURES_DIR" '.data.chunks_dir = $p' "$SCRATCH/run/config.json" > "$SCRATCH/run/config.patched.json"
mv "$SCRATCH/run/config.patched.json" "$SCRATCH/run/config.json"

# Optional: show a couple of example files
echo "[INFO] Example feature files:"
find "$FEATURES_DIR" -type f -name "*.pt" | head -n 5

# Run training from SSD
echo "[STEP] Start training..."
cd "$SCRATCH/run"
python "$SLURM_SUBMIT_DIR/$TRAIN_SCRIPT" --config "$SCRATCH/run/config.json" -v

# Collect outputs
JOB_OUT="$OUT_BASE/${SLURM_JOB_ID}"
mkdir -p "$JOB_OUT"
echo "[STEP] Copy results back to shared disk: $JOB_OUT"
rsync -a "$SCRATCH/run/" "$JOB_OUT/run/"
rsync -a "$SCRATCH/features/" "$JOB_OUT/features_copy/" --delete  # optional snapshot of what we trained on
echo "[DONE] Outputs saved to $JOB_OUT"
