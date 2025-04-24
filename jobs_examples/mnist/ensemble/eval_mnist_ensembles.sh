#!/bin/bash

#SBATCH --job-name=eval_model_array
#SBATCH --partition=gpu_short 
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --account=coms030646
#SBATCH --array=0-1  # Adjust based on the number of models

# # Define model types and trained model paths as space-separated strings
ENSEMBLE_CONFIG_NAMES=("mnist_ensemble_mamba" "mnist_ensemble_mamba_orig")
TRAINED_ENSEMBLE_PATHS=("/user/work/js21767/Project/out/mnist/ensemble/mnist_ensemble_mamba/2025_03_14_PM02_32_ensemble/ensemble_acc_100.0.pth"
                        "/user/work/js21767/Project/out/mnist/ensemble/mnist_ensemble_mamba_orig/2025_03_14_PM02_32_ensemble/ensemble_acc_100.0.pth")
ENSEMBLE_CONFIG_NAME=${ENSEMBLE_CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}
TRAINED_ENSEMBLE_PATH=${TRAINED_ENSEMBLE_PATHS[$SLURM_ARRAY_TASK_ID]}
# ENSEMBLE_CONFIG_NAME="mnist_ensemble_swin_orig"
# TRAINED_ENSEMBLE_PATH="/user/work/js21767/Project/out/mnist/ensemble/mnist_ensemble_swin_orig/2025_03_13_PM09_16_ensemble/ensemble_acc_100.0.pth"

# Paths
ENSEMBLE_CONFIG_PATH="/user/home/js21767/Project/my_configs/models_train/ensemble/$ENSEMBLE_CONFIG_NAME.yml"
OUT_DIR="/user/work/js21767/Project/out/mnist/ensemble/$ENSEMBLE_CONFIG_NAME/"
DATASET_CONFIG_PATH="my_configs/datasets/bloodmnist.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run evaluation script, passing entire lists
python3 eval_ensemble.py --out_dir "$OUT_DIR" \
                --ensemble_config_path $ENSEMBLE_CONFIG_PATH \
                --dataset_config_path "$DATASET_CONFIG_PATH" \
                --dataset_download_dir "$DATASET_DOWNLOAD_DIR" \
                --stacking_model_path $TRAINED_ENSEMBLE_PATH \

