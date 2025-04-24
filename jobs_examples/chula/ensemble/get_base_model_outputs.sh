#!/bin/bash

#SBATCH --job-name=train_model_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --account=coms030646
#SBATCH --array=0-5  # Adjust this based on the number of models

# Define the model configuration paths
MODELS=("localmamba" "mambavision" "medmamba" "swin" "vim" "vmamba")
MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
# MODEL_NAME="medmamba"

echo "Training model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/chula/ensemble/base_model_outputs/"
MODEL_CONFIG_PATH="my_configs/models_train/${MODEL_NAME}.yml"
DATASET_CONFIG_PATH="my_configs/datasets/chula_augmented.yml"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run training script
python3 train.py --out_dir="$OUT_DIR" \
                 --model_config_path="$MODEL_CONFIG_PATH" \
                 --dataset_config_path="$DATASET_CONFIG_PATH" \
                 --num_folds=-5 \
                 --num_gpus=1 \
                 --verbose \