#!/bin/bash

#SBATCH --job-name=eval_model_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=coms030646
#SBATCH --array=0-5  # Adjust based on the number of models

# Define the model configuration paths
MODELS=("localmamba" "mambavision" "medmamba" "swin" "vim" "vmamba")
TRAINED_MODEL_PATHS=("/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM12_55_localmamba/localmamba_fold_2_acc_99.1638795986622.pth" 
                     "/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM12_55_mambavision/mambavision_fold_2_acc_99.28929765886288.pth"
                     "/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM11_40_medmamba/medmamba_fold_2_acc_98.16053511705685.pth"
                     "/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM12_55_swin/swin_fold_2_acc_99.08026755852842.pth"
                     "/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM01_00_vim/vim_fold_2_acc_98.82943143812709.pth"
                     "/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM12_55_vmamba/vmamba_fold_2_acc_98.99665551839465.pth")

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
TRAINED_MODEL_PATH=${TRAINED_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

# Debugging output
echo "Evaluating model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/mnist/no_aug/eval/"
DATASET_CONFIG_PATH="my_configs/datasets/bloodmnist.yml"
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
