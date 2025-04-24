#!/bin/bash

#SBATCH --job-name=train_model_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:30:00
#SBATCH --account=coms030646
# #SBATCH --array=0-5

# # Define the model configuration paths
# MODELS=("localmamba" "vmamba" "mambavision" "medmamba" "swin" "vim")
# MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="medmamba"
echo "Training model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/mnist/no_aug/"
MODEL_CONFIG_PATH="my_configs/models_train/${MODEL_NAME}.yml"
DATASET_CONFIG_PATH="my_configs/datasets/bloodmnist.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run training script
python3 train.py --out_dir="$OUT_DIR" \
                 --model_config_path="$MODEL_CONFIG_PATH" \
                 --dataset_config_path="$DATASET_CONFIG_PATH" \
                 --num_gpus=1 \
                 --verbose \
                 --dataset_download_dir="$DATASET_DOWNLOAD_DIR"
