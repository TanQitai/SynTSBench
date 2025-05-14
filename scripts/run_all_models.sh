#!/bin/bash

# Trap to catch interrupt and terminate signals and kill child processes
trap cleanup INT TERM EXIT

# Keep track of all spawned PIDs
declare -a PIDS

cleanup() {
    echo "Caught signal - cleaning up processes..."
    # Kill all processes in our process group
    for pid in "${PIDS[@]}"; do
        echo "Killing process $pid and its children"
        pkill -P $pid 2>/dev/null || true
        kill -9 $pid 2>/dev/null || true
    done
    
    # Make sure no python processes remain from our script
    pkill -f "python -u run.py" || true
    
    echo "Cleanup complete."
    exit 1
}

# Configuration
MAX_CONCURRENT_JOBS=2  # Maximum number of concurrent jobs per GPU
MODELS=("TimeKAN" "TimeMixer" "TimesNet" "TSMixer" "SegRNN" "PatchTST" "iTransformer" "DLinear" "Autoformer" "PaiFilter" "TexFilter" )  # List of models to run
GPU_ASSIGNMENTS=("0" "0" "1" "1" "2" "2" "1" "2" "3" "3" "0" )  # GPU assignments for each model

# Create logs directory
mkdir -p logs

# Function to run a job
run_job() {
    local model=$1
    local gpu=$2
    local log_file="logs/${model}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting $model on GPU $gpu at $(date)"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/${model}.sh > $log_file 2>&1 &
    local pid=$!
    PIDS+=($pid)  # Store PID in array for cleanup
    echo "$model (PID $pid) started on GPU $gpu"
}

# Track running jobs per GPU
declare -A running_jobs_per_gpu
for gpu in $(echo "${GPU_ASSIGNMENTS[@]}" | tr ' ' '
' | sort -u); do
    running_jobs_per_gpu[$gpu]=0
done

# Process all models
for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    gpu=${GPU_ASSIGNMENTS[$i]}
    
    # Wait if maximum concurrent jobs reached for this GPU
    while [ ${running_jobs_per_gpu[$gpu]} -ge $MAX_CONCURRENT_JOBS ]; do
        echo "Waiting for a slot on GPU $gpu (currently running ${running_jobs_per_gpu[$gpu]} jobs)..."
        
        # More robust job counting
        for job_gpu in "${!running_jobs_per_gpu[@]}"; do
            # Count active jobs for this GPU
            running_count=0
            for pid in "${PIDS[@]}"; do
                if ps -p "$pid" > /dev/null; then
                    running_count=$((running_count + 1))
                fi
            done
            running_jobs_per_gpu[$job_gpu]=$running_count
            echo "GPU $job_gpu has $running_count active jobs"
        done
        
        if [ ${running_jobs_per_gpu[$gpu]} -lt $MAX_CONCURRENT_JOBS ]; then
            break
        fi
        
        sleep 10  # Longer sleep to reduce CPU usage
    done
    
    # Run the model script
    run_job $model $gpu
    sleep 5  # Add small delay between job submissions
    running_jobs_per_gpu[$gpu]=$((running_jobs_per_gpu[$gpu] + 1))
done

# Wait for all jobs to complete
echo "All jobs submitted. Waiting for completion..."
wait

echo "All model executions completed!"
