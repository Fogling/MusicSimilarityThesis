#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=64G
#SBATCH --tmp=100G
#SBATCH --time=1-00:00:00

# Exit on error or undefined variables
set -euo pipefail

# These variables are passed in via --export from batch_experiments.sh:
# - CONFIG_JSON: Path to the config file for this job
# - ARCHIVE: Path to the data archive
# - TRAIN_PY: Path to the training script
# - OUT_BASE: Base directory for results
# - CONFIG_NAME: Name of the config (for organized output)

echo "================================================================"
echo "🧪 Batch Experiment Job"
echo "================================================================"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_JSON}"
echo "Config Name: ${CONFIG_NAME}"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Temp Dir: ${SLURM_JOB_TMP}"
echo "================================================================"

# CPU/GPU threading balance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

start_time=$(date +%s)

# Stage data on the node-local SSD
echo "📦 Staging archive to node SSD..."
cp -v "${ARCHIVE}" "${SLURM_JOB_TMP}/"

echo "📂 Unpacking on SSD..."
cd "${SLURM_JOB_TMP}"

fname="$(basename "${ARCHIVE}")"
if [[ "${fname}" == *.zip ]]; then
    unzip -q "${fname}"
elif [[ "${fname}" == *.tar.gz || "${fname}" == *.tgz ]]; then
    tar -xzf "${fname}"
else
    echo "❌ ERROR: Unsupported archive type: ${fname}" >&2
    exit 1
fi

echo "✅ Unpacked. Top-level contents:"
find . -maxdepth 2 -type d | sed 's#^\./##'

# Verify config file exists and is readable
if [[ ! -f "$CONFIG_JSON" ]]; then
    echo "❌ ERROR: Config file not found: $CONFIG_JSON" >&2
    exit 1
fi

echo "⚙️  Config file contents:"
echo "---"
head -20 "$CONFIG_JSON" | sed 's/^/  /'
echo "---"

# Run training
echo "🚀 Starting training with config: $CONFIG_NAME"
echo "Working directory: $(pwd)"

# Run the training and capture exit code
TRAINING_EXIT_CODE=0
python "${TRAIN_PY}" --config "${CONFIG_JSON}" || TRAINING_EXIT_CODE=$?

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
fi

# Create organized output directory for this specific config
CONFIG_RESULTS_DIR="${OUT_BASE}/${CONFIG_NAME}_${SLURM_JOB_ID}"
mkdir -p "${CONFIG_RESULTS_DIR}"

echo "📊 Collecting results to: ${CONFIG_RESULTS_DIR}"

# Copy training results
shopt -s nullglob
for pattern in \
    run_* \
    *.log \
    *.json
do
    for p in $pattern; do
        if [[ -e "$p" ]]; then
            echo "  ➡️  Copying $p"
            cp -r "$p" "${CONFIG_RESULTS_DIR}/"
        fi
    done
done

# Copy non-empty directories
for dir in ast_triplet_output logs; do
    if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -type f 2>/dev/null)" ]]; then
        echo "  ➡️  Copying $dir (non-empty)"
        cp -r "$dir" "${CONFIG_RESULTS_DIR}/"
    elif [[ -d "$dir" ]]; then
        echo "  ⚠️  Skipping $dir (empty)"
    fi
done
shopt -u nullglob

# Create a job info file with metadata
JOB_INFO_FILE="${CONFIG_RESULTS_DIR}/job_info.json"
cat > "$JOB_INFO_FILE" << EOF
{
    "job_id": "${SLURM_JOB_ID}",
    "job_name": "${SLURM_JOB_NAME}",
    "config_name": "${CONFIG_NAME}",
    "config_file": "${CONFIG_JSON}",
    "node": "$(hostname)",
    "start_time": "$(date -d @$start_time --iso-8601=seconds)",
    "end_time": "$(date --iso-8601=seconds)",
    "runtime_seconds": $(($(date +%s) - start_time)),
    "training_exit_code": $TRAINING_EXIT_CODE,
    "status": $(if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then echo '"SUCCESS"'; else echo '"FAILED"'; fi)
}
EOF

# Copy the actual config file used
cp "${CONFIG_JSON}" "${CONFIG_RESULTS_DIR}/config_used.json"

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "================================================================"
echo "📈 Job Summary"
echo "================================================================"
echo "Config: ${CONFIG_NAME}"
echo "Status: $(if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then echo "✅ SUCCESS"; else echo "❌ FAILED"; fi)"
echo "Runtime: $runtime seconds ($(date -ud @$runtime +%H:%M:%S))"
echo "Results: ${CONFIG_RESULTS_DIR}"
echo "================================================================"

# Exit with the same code as training
exit $TRAINING_EXIT_CODE