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
ARCHIVE="$HOME/thesis/precomputed_AST.zip"          # .zip or .tar.gz both ok
CONFIG_JSON="$HOME/thesis/train_from_precomputed.json"
TRAIN_PY="$HOME/thesis/AST_Triplet_training.py"
OUT_BASE="$HOME/thesis/runs"                        # where results will be copied
################################

# 1) Friendly info
echo "[INFO] SLURM_JOB_TMP is: ${SLURM_JOB_TMP}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES:-unset}"

# CPU/GPU balance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

start_time=$(date +%s)

# 3) Stage data on the node-local SSD
echo "[INFO] Staging archive to node SSD..."
cp -v "${ARCHIVE}" "${SLURM_JOB_TMP}//"

echo "[INFO] Unpacking on SSD..."
cd "${SLURM_JOB_TMP}"

fname="$(basename "${ARCHIVE}")"
if [[ "${fname}" == *.zip ]]; then
  unzip -q "${fname}"
elif [[ "${fname}" == *.tar.gz || "${fname}" == *.tgz ]]; then
  tar -xzf "${fname}"
else
  echo "[ERROR] Unsupported archive type: ${fname}" >&2
  exit 1
fi
echo "[OK] Unpacked. Top-level contents:"
find . -maxdepth 2 -type d | sed 's#^\./##'

# 4) Run training (WORKING DIR = SSD)
#    Your config uses data.chunks_dir="precomputed_AST". Because we are now
#    in $SLURM_JOB_TMP and we unpacked here, the relative path resolves
#    correctly. Inside Python you can also do: cache = os.environ["SLURM_JOB_TMP"].
echo "[INFO] Starting training..."
python "${TRAIN_PY}" --config "${CONFIG_JSON}"

# 5) (Optional) Collect results back to $HOME in a run-specific folder
RUN_DIR="${OUT_BASE}/${SLURM_JOB_NAME}.${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

echo "[INFO] Collecting artifacts to ${RUN_DIR} ..."
# trainer outputs (if any), split info, final model folders, logs
shopt -s nullglob
for p in \
  fully_trained_models_* \
  splits_* \
  ast_triplet_output \
  logs \
  *.log
do
  if [[ -e "$p" ]]; then
    echo "  -> copying $p"
    cp -r "$p" "${RUN_DIR}/"
  fi
done
shopt -u nullglob

echo "[DONE] Everything copied to: ${RUN_DIR}"

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job runtime: $runtime seconds ($(date -ud @$runtime +%H:%M:%S))"