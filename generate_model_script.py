#!/usr/bin/env python3
# filepath: /fs-computility/mllm1/limo/workspace/tanqitai/Time-Series-Library-main/generate_model_script_revise.py

import os
import argparse
import json
from typing import Dict, List, Any, Optional
import subprocess

# Base configuration for all models
BASE_CONFIG = {
    "dataset_dir": "../dataset/electricity",  # 修改为Exchange数据集路径
    "file_pattern": "electricity.csv",  # 修改为Exchange数据集文件名
    "seq_len": 96,
    "pred_lens": [24, 48, 96, 192],  # 支持多个预测长度
    "is_training": 1,
    "features": "M",  # 修改为多变量特征
    "data": "custom",  # 修改为自定义数据类型
    "enc_in": 321,  # 修改为输入维度
    "dec_in": 321,  # 修改为解码器输入维度
    "c_out": 321,  # 修改为输出维度
    "des": "Exp",
    "itr": 1,
    "train_epochs": 5,
    # "inverse": True,
}

# Model-specific configurations (will override base config)
MODEL_CONFIGS = {
    "TimeKAN": {
        "e_layers": 2,
        "d_model": 16,
        "d_ff": 32,
        "learning_rate": 0.01,
        "train_epochs": 10,
        "patience": 10,
        "batch_size": 128,
        "down_sampling_layers": 2,
        "down_sampling_window": 2,
        "begin_order": 0,
    },
    "TimeMixer": {
        "e_layers": 2,
        "d_model": 16,
        "d_ff": 32,
        "learning_rate": 0.01,
        "train_epochs": 10,
        "patience": 10,
        "batch_size": 128,
        "down_sampling_layers": 2,
        "down_sampling_method": "avg",
        "down_sampling_window": 2,
    },
    "TimesNet": {
        "e_layers": 2,
        "d_layers": 1,
        "factor": 3,
        "d_model": 16,
        "d_ff": 32,
        "top_k": 5,
    },
    "TSMixer": {
        "e_layers": 2,
        "d_layers": 1,
        "factor": 3,
    },
    "SegRNN": {
        "seg_len": 2,
        "d_model": 512,
        "dropout": 0.5,
        "learning_rate": 0.0001,
    },
    "PatchTST": {
        "e_layers": 3,
        "d_layers": 1,
        "factor": 3,
        "patch_len": 16,
        "n_heads": 4,
        "learning_rate": 0.0001,
    },
    "iTransformer": {
        "e_layers": 3,  # Number of encoder layers
        "d_model": 512,  # Model dimension
        "d_ff": 512,  # Feed-forward network dimension
        "n_heads": 8,  # Number of attention heads
        "dropout": 0.1,  # Dropout rate
        "learning_rate": 0.0005,  # Learning rate
        "factor": 3,  # Attention factor
    },
    "DLinear": {
        "e_layers": 2,
        "d_layers": 1,
        "factor": 3,
        "label_len": 10,  # Override base config
        "learning_rate": 0.01,
    },
    "Autoformer": {
        "e_layers": 2,
        "d_layers": 1,
        "factor": 3,
        "label_len": 10,  # Override base config
        "moving_avg": 25,
        "learning_rate": 0.001,
    },
    "PaiFilter": {
        "hidden_size": 256,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 10,
        "learning_rate": 0.01,
    },
    "TexFilter": {
        "embed_size": 512,
        "hidden_size": 512,
        "dropout": 0,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 10,
        "learning_rate": 0.001,
    },
}

# GPU assignments
GPU_ASSIGNMENTS = {
    "TimeKAN": "1",
    "TimeMixer": "2",
    "TimesNet": "1",
    "TSMixer": "1",
    "iTransformer": "1",
    "SegRNN": "2",
    "PatchTST": "2", 
    "DLinear": "2",
    "Autoformer": "3",
    "PaiFilter": "3",
    "TexFilter": "3",
}


def generate_script(model_name: str, config: Dict[str, Any], output_dir: str) -> str:
    """Generate a shell script for a specific model with the given configuration."""
    script_content = f"""#!/bin/bash

model_name={model_name}
"""
    
    # Add model-specific parameter declarations
    for key, value in config.items():
        if key not in ["is_training", "features", "data", "enc_in", "dec_in", "c_out", 
                      "des", "itr", "inverse", "dataset_dir", "file_pattern", "pred_lens"]:
            script_content += f"{key}={value}\n"
    
    script_content += f"""
# Path to your dataset directory
DATASET_DIR="{config['dataset_dir']}"
# Path to checkpoints directory
CHECKPOINTS_DIR="./checkpoints"

# Find datasets matching the pattern
CSV_FILES=$(find ${{DATASET_DIR}} -name "{config['file_pattern']}")

# Define prediction lengths to test
PRED_LENS=({' '.join(str(pl) for pl in config['pred_lens'])})

# Process each CSV file
for csv_file in ${{CSV_FILES}}; do
    # Get just the filename without path
    filename=$(basename "$csv_file")
    
    # Loop through each prediction length
    for pred_len in "${{PRED_LENS[@]}}"; do
        # Create a model_id based on the filename and prediction length
        model_id="${{filename%.csv}}_${{seq_len}}_${{pred_len}}_{model_name}"
        
        # # Check if this model_id has already been processed
        # # Look for checkpoint directory that matches our model_id
        # if ls ${{CHECKPOINTS_DIR}}/long_term_forecast_${{model_id}}_* 1> /dev/null 2>&1; then
        #     echo "Skipping $filename with pred_len=${{pred_len}} - already processed (found checkpoint)"
        #     continue
        # fi
        
        echo "Processing file: $filename with pred_len=${{pred_len}}"
        echo "Model ID: $model_id"
        
        python -u run.py \\
          --task_name long_term_forecast \\
          --is_training {config['is_training']} \\
          --root_path "${{DATASET_DIR}}/" \\
          --data_path "$filename" \\
          --model_id "$model_id" \\
          --model $model_name \\
          --data {config['data']} \\
          --features {config['features']} \\
          --seq_len ${{seq_len}} \\
          --label_len {config.get('label_len', 0)} \\
          --pred_len ${{pred_len}} \\
"""

    # Add all the model-specific parameters
    for key, value in config.items():
        if key not in ["seq_len", "pred_lens", "is_training", "features", "data", "label_len",
                    "dataset_dir", "file_pattern", "inverse"]:
            script_content += f"          --{key} {value} \\\n"

    # Add inverse as a flag (no value) if needed
    if config.get("inverse", False):
        script_content += "          --inverse\n"
    else:
        script_content = script_content[:-2] + "\n"  # Remove trailing backslash
    
    script_content += """        
        echo "Completed processing $filename with pred_len=$pred_len"
        echo "---------------------------------"
    done
done

echo "All files processed!"
"""
    
    # Save the script
    script_path = os.path.join(output_dir, f"{model_name}.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    return script_path

def generate_run_all_script(models: List[str], gpu_assignments: Dict[str, str], output_dir: str) -> str:
    """Generate a script that runs all model scripts with GPU assignments."""
    # The run_all_script function remains mostly unchanged
    # ... existing code for generate_run_all_script ...
    script_content = """#!/bin/bash

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
MODELS=("""

    for model in models:
        script_content += f'"{model}" '
    
    script_content += """)  # List of models to run
GPU_ASSIGNMENTS=("""
    
    for model in models:
        script_content += f'"{gpu_assignments[model]}" '
    
    script_content += """)  # GPU assignments for each model

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
for gpu in $(echo "${GPU_ASSIGNMENTS[@]}" | tr ' ' '\n' | sort -u); do
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
"""
    
    script_path = os.path.join(output_dir, "run_all_models.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description='Generate or update model scripts with unified parameters')
    parser.add_argument('--seq_len', type=int, help='Input sequence length')
    parser.add_argument('--pred_lens', type=str, help='Comma-separated prediction lengths (e.g., 96,192,336)')
    parser.add_argument('--file_pattern', type=str, help='File pattern to match')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--save_config', type=str, help='Save current config to JSON file')
    parser.add_argument('--load_config', type=str, help='Load config from JSON file')
    parser.add_argument('--output_dir', type=str, default='scripts', help='Output directory for scripts')
    parser.add_argument('--run', action='store_true', help='Run the scripts after generating them')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize with base config
    configs = {}
    for model_name, model_config in MODEL_CONFIGS.items():
        configs[model_name] = BASE_CONFIG.copy()
        configs[model_name].update(model_config)
    
    # Load config if provided
    if args.load_config:
        with open(args.load_config, 'r') as f:
            loaded_configs = json.load(f)
        for model_name, loaded_config in loaded_configs.items():
            if model_name in configs:
                configs[model_name].update(loaded_config)
    
    # Update configs with command-line arguments
    for model_name in configs:
        if args.seq_len:
            configs[model_name]['seq_len'] = args.seq_len
        if args.pred_lens:
            # Convert comma-separated string to list of integers
            configs[model_name]['pred_lens'] = [int(x.strip()) for x in args.pred_lens.split(',')]
        if args.file_pattern:
            configs[model_name]['file_pattern'] = args.file_pattern
        if args.dataset_dir:
            configs[model_name]['dataset_dir'] = args.dataset_dir
    
    # Save config if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(configs, f, indent=2)
    
    # Generate scripts with built-in resume capability
    model_scripts = []
    for model_name, config in configs.items():
        script_path = generate_script(model_name, config, args.output_dir)
        model_scripts.append(script_path)
        print(f"Generated script for {model_name}: {script_path}")
    
    # Generate run_all_models script
    run_all_script = generate_run_all_script(list(MODEL_CONFIGS.keys()), GPU_ASSIGNMENTS, args.output_dir)
    print(f"Generated run all script: {run_all_script}")
    
    # Run if requested
    if args.run:
        subprocess.run(['bash', run_all_script])

if __name__ == "__main__":
    main()