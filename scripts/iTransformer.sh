#!/bin/bash

model_name=iTransformer
seq_len=96
pred_len=96
train_epochs=30
label_len=0
e_layers=3
d_model=512
d_ff=512
n_heads=8
dropout=0.1
learning_rate=0.0005
factor=3

# Path to your dataset directory
DATASET_DIR="../dataset/Dataset_generated_anomaly"
# Path to checkpoints directory
CHECKPOINTS_DIR="./checkpoints"

# List of signal types to process
SIGNAL_TYPES=("Double-Sin" "Five-Sin" "Linear-Trend" "Logistic-Trend" "Quadratic-Function" "Sin0.05")

# List of noise levels to process
NOISE_LEVELS=("point-anomaly-15pct" "point-anomaly-20pct" "pulse-anomaly-10" "pulse-anomaly-20")

# List of dataset lengths to process
DATASET_LENGTHS=("5000")

# Process each signal type, noise level, and dataset length
for signal_type in "${SIGNAL_TYPES[@]}"; do
    for noise_level in "${NOISE_LEVELS[@]}"; do
        for dataset_length in "${DATASET_LENGTHS[@]}"; do
            # Find datasets matching the pattern for this signal type, noise level, and length
            pattern="dataset*_${signal_type}_${noise_level}_length${dataset_length}.csv"
            echo "Looking for files matching: $pattern"
            
            CSV_FILES=$(find ${DATASET_DIR} -name "$pattern")
            
            # Process each CSV file
            for csv_file in ${CSV_FILES}; do
                # Get just the filename without path
                filename=$(basename "$csv_file")
                
                # Create a model_id based on the filename
                model_id="${filename%.csv}_${seq_len}_${pred_len}_iTransformer"
                
                # # Check if this model_id has already been processed
                # if ls ${CHECKPOINTS_DIR}/long_term_forecast_${model_id}_* 1> /dev/null 2>&1; then
                #     echo "Skipping $filename - already processed (found checkpoint)"
                #     continue
                # fi
                
                echo "Processing file: $filename"
                echo "Model ID: $model_id"
                
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path "${DATASET_DIR}/" \
                  --data_path "$filename" \
                  --model_id "$model_id" \
                  --model $model_name \
                  --data generated \
                  --features S \
                  --seq_len ${seq_len} \
                  --label_len 0 \
                  --pred_len ${pred_len} \
              --train_epochs 30 \
              --enc_in 1 \
              --dec_in 1 \
              --c_out 1 \
              --des Exp \
              --itr 1 \
              --e_layers 3 \
              --d_model 512 \
              --d_ff 512 \
              --n_heads 8 \
              --dropout 0.1 \
              --learning_rate 0.0005 \
              --factor 3 
            
                echo "Completed processing $filename"
                echo "---------------------------------"
            done
        done
    done
done

echo "All files processed!"
