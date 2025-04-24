#!/bin/bash

#SBATCH --job-name=eval_model_array
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=coms030646
#SBATCH --array=0-5  # Adjust based on the number of models

# Define the model configuration paths
MODELS=("localmamba" "mambavision" "medmamba" "swin" "vim" "vmamba")
TRAINED_MODEL_PATHS=("/user/work/js21767/Project/out/mnist/aug/2025_03_13_PM07_26_localmamba/localmamba_fold_2_acc_99.12333736396614.pth"
                     "/user/work/js21767/Project/out/mnist/aug/2025_03_13_PM07_26_mambavision/mambavision_fold_2_acc_99.24425634824668.pth"
                     "/user/work/js21767/Project/out/mnist/aug/2025_03_13_PM07_26_medmamba/medmamba_fold_2_acc_96.85610640870617.pth"
                     "/user/work/js21767/Project/out/mnist/aug/2025_03_13_PM07_26_swin/swin_fold_2_acc_99.2744860943168.pth"
                     "/user/work/js21767/Project/out/mnist/aug/2025_03_13_PM07_26_vim/vim_fold_2_acc_99.00241837968561.pth"
                     "/user/work/js21767/Project/out/mnist/aug/2025_03_13_PM07_26_vmamba/vmamba_fold_2_acc_99.21402660217655.pth"
                     )

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
TRAINED_MODEL_PATH=${TRAINED_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

# Debugging output
echo "Evaluating model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/mnist/aug/eval/"
DATASET_CONFIG_PATH="my_configs/datasets/bloodmnist_augmented.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run evaluation script
python3 eval.py --trained_model_path="$TRAINED_MODEL_PATH" \
                --model_type="$MODEL_NAME" \
                --dataset_config_path="$DATASET_CONFIG_PATH" \
                --dataset_download_dir="$DATASET_DOWNLOAD_DIR"
