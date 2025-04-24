#!/bin/bash

#SBATCH --job-name=train_model_array
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --account=coms030646

# Paths
OUT_DIR="/user/work/js21767/Project/out/chula/hybrid/pre_trained"
MODEL_CONFIG_PATH="my_configs/models_train/hybrid/hybrid_train.yml"
DATASET_CONFIG_PATH="my_configs/datasets/chula_hybrid.yml"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run training script
python3 train.py --out_dir="$OUT_DIR" \
                 --model_config_path="$MODEL_CONFIG_PATH" \
                 --dataset_config_path="$DATASET_CONFIG_PATH" \
                 --num_folds=1 \
                 --num_gpus=1 \
                 --verbose
