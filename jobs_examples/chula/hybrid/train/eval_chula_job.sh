#!/bin/bash

#SBATCH --job-name=eval_model_array
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=coms030646

TRAINED_MODEL_PATH="/user/work/js21767/Project/out/chula/hybrid/fine_tuned/2025_04_24_AM10_55_hybrid/hybrid_acc_84.50657894736841.pth"

# Paths
OUT_DIR="/user/work/js21767/Project/out/chula/hybrid/"
DATASET_CONFIG_PATH="my_configs/datasets/chula.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"
MODEL_CONFIG="my_configs/models_train/hybrid/hybrid_fine_tune.yml"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run evaluation script
python3 eval.py --trained_model_path="$TRAINED_MODEL_PATH" \
                --model_config="$MODEL_CONFIG" \
                --dataset_config_path="$DATASET_CONFIG_PATH" \
                --dataset_download_dir="$DATASET_DOWNLOAD_DIR"
