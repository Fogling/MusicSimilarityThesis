#!/bin/bash

# Batch Experiment Runner
# Submits one SLURM job for each config file found in the configs directory
# Usage: ./batch_experiments.sh [config_directory] [max_concurrent_jobs]

set -euo pipefail

######### CONFIGURATION #########
CONFIGS_DIR="${1:-configs/experiments}"      # Directory containing config files
MAX_CONCURRENT="${2:-4}"                     # Maximum concurrent jobs
ARCHIVE="$HOME/thesis/precomputed_AST_7G.zip" # Data archive (same as original)
TRAIN_PY="$HOME/thesis/AST_Triplet_training.py" # Training script
OUT_BASE="$HOME/thesis/batch_runs"          # Base directory for all batch results
##################################

echo "================================================================"
echo "🚀 Batch Experiment Runner"
echo "================================================================"
echo "Config directory: $CONFIGS_DIR"
echo "Max concurrent jobs: $MAX_CONCURRENT"
echo "Archive: $ARCHIVE"
echo "Results will be saved to: $OUT_BASE"
echo "================================================================"

# Check if configs directory exists
if [[ ! -d "$CONFIGS_DIR" ]]; then
    echo "❌ Error: Config directory '$CONFIGS_DIR' does not exist!"
    echo "Create it and add your config files, or specify a different directory:"
    echo "Usage: $0 [config_directory] [max_concurrent_jobs]"
    exit 1
fi

# Find all JSON config files
mapfile -t CONFIG_FILES < <(find "$CONFIGS_DIR" -name "*.json" -type f | sort)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "❌ No .json config files found in '$CONFIGS_DIR'"
    exit 1
fi

echo "📁 Found ${#CONFIG_FILES[@]} config files:"
for config in "${CONFIG_FILES[@]}"; do
    echo "  - $(basename "$config")"
done
echo

# Create batch results directory with timestamp
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_RESULTS_DIR="$OUT_BASE/batch_$BATCH_TIMESTAMP"
mkdir -p "$BATCH_RESULTS_DIR"

echo "📊 Batch results will be collected in: $BATCH_RESULTS_DIR"
echo

# Create a summary file to track all submitted jobs
BATCH_SUMMARY="$BATCH_RESULTS_DIR/batch_summary.txt"
echo "# Batch Experiment Summary - $BATCH_TIMESTAMP" > "$BATCH_SUMMARY"
echo "# Config File -> Job ID -> Status" >> "$BATCH_SUMMARY"
echo >> "$BATCH_SUMMARY"

# Function to count running jobs for this batch
count_running_jobs() {
    squeue -u "$USER" --name="batch-exp-*" --noheader 2>/dev/null | wc -l
}

# Function to wait for job slot
wait_for_slot() {
    while [[ $(count_running_jobs) -ge $MAX_CONCURRENT ]]; do
        echo "⏳ Waiting for job slot ($(count_running_jobs)/$MAX_CONCURRENT jobs running)..."
        sleep 30
    done
}

# Submit jobs for each config file
JOB_COUNTER=0
SUBMITTED_JOBS=()

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    JOB_COUNTER=$((JOB_COUNTER + 1))
    
    # Extract config name for job naming
    CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
    CONFIG_GROUP=$(basename "$(dirname "$CONFIG_FILE")")
    
    # Create a clean job name (SLURM has character limits)
    if [[ "$CONFIG_GROUP" == "experiments" ]]; then
        JOB_NAME="batch-exp-$CONFIG_NAME"
    else
        JOB_NAME="batch-exp-${CONFIG_GROUP}-$CONFIG_NAME"
    fi
    
    # Ensure job name isn't too long (SLURM limit is usually 64 chars)
    JOB_NAME="${JOB_NAME:0:60}"
    
    echo "🎯 [$JOB_COUNTER/${#CONFIG_FILES[@]}] Submitting: $CONFIG_NAME"
    echo "   Config: $CONFIG_FILE"
    echo "   Job name: $JOB_NAME"
    
    # Wait for available job slot
    wait_for_slot
    
    # Submit the job using our template script
    JOB_OUTPUT=$(sbatch \
        --job-name="$JOB_NAME" \
        --export=CONFIG_JSON="$CONFIG_FILE",ARCHIVE="$ARCHIVE",TRAIN_PY="$TRAIN_PY",OUT_BASE="$BATCH_RESULTS_DIR",CONFIG_NAME="$CONFIG_NAME" \
        batch_job_template.sbatch)
    
    # Extract job ID from sbatch output
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+$')
    
    echo "   ✅ Job ID: $JOB_ID"
    echo
    
    # Record in summary
    echo "$CONFIG_FILE -> $JOB_ID -> SUBMITTED" >> "$BATCH_SUMMARY"
    SUBMITTED_JOBS+=("$JOB_ID:$CONFIG_NAME")
    
    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo "================================================================"
echo "🎉 Batch submission complete!"
echo "================================================================"
echo "Submitted ${#CONFIG_FILES[@]} jobs with IDs:"
for job_info in "${SUBMITTED_JOBS[@]}"; do
    IFS=':' read -r job_id config_name <<< "$job_info"
    echo "  - $config_name: $job_id"
done

echo
echo "📊 Monitor progress with:"
echo "   watch 'squeue -u $USER --name=\"batch-exp-*\"'"
echo
echo "📁 Results will be collected in:"
echo "   $BATCH_RESULTS_DIR"
echo
echo "📋 Job summary saved to:"
echo "   $BATCH_SUMMARY"

# Create a helper script for monitoring
MONITOR_SCRIPT="$BATCH_RESULTS_DIR/monitor_batch.sh"
cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
echo "🔍 Monitoring batch jobs..."
echo "Active jobs:"
squeue -u $USER --name="batch-exp-*" --format="%.10i %.20j %.8T %.10M %.10L %.6D %R"
echo
echo "📊 Job summary:"
cat "$BATCH_SUMMARY"
echo
echo "📁 Results directory: $BATCH_RESULTS_DIR"
EOF
chmod +x "$MONITOR_SCRIPT"

echo "🔍 Monitor script created: $MONITOR_SCRIPT"
echo
echo "================================================================"
echo "Happy experimenting! 🧪"
echo "================================================================"