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
TRAINED_MODEL_PATHS=("/user/work/js21767/Project/out/chula/pretrain_aug/2025_02_26_PM03_52_localmamba/localmamba_fold_2.pth"
                     "/user/work/js21767/Project/out/chula/pretrain_aug/2025_02_26_PM05_07_mambavision/mambavision_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/pretrain_aug/2025_02_26_PM05_30_medmamba/medmamba_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/pretrain_aug/2025_02_26_PM04_28_swin/swin_fold_4.pth"
                     "/user/work/js21767/Project/out/chula/pretrain_aug/2025_02_26_PM05_03_vim/vim_fold_4.pth"
                     "/user/work/js21767/Project/out/chula/pretrain_aug/2025_02_26_PM04_21_vmamba/vmamba_fold_3.pth"
                     )
NEUTROPHIL_MODEL_PATHS=("/user/work/js21767/Project/out/chula/neutrophils/2025_03_02_PM05_27_localmamba/localmamba_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/neutrophils/2025_03_02_PM03_57_mambavision/mambavision_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/neutrophils/2025_03_02_PM04_01_medmamba/medmamba_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/neutrophils/2025_03_02_PM03_31_swin/swin_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/neutrophils/2025_03_02_PM04_04_vim/vim_fold_3.pth"
                     "/user/work/js21767/Project/out/chula/neutrophils/2025_03_02_PM04_48_vmamba/vmamba_fold_3.pth")

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
TRAINED_MODEL_PATH=${TRAINED_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
NEUTROPHIL_MODEL_PATH=${NEUTROPHIL_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

# Debugging output
echo "Evaluating model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/chula/neutrophils/eval/"

DATASET_CONFIG_PATH="my_configs/datasets/chula.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run evaluation script
python3 eval.py --out_dir="$OUT_DIR" \
                --trained_model_path="$TRAINED_MODEL_PATH" \
                --neutrophil_model_path="$NEUTROPHIL_MODEL_PATH" \
                --model_type="$MODEL_NAME" \
                --dataset_config_path="$DATASET_CONFIG_PATH" \
                --dataset_download_dir="$DATASET_DOWNLOAD_DIR"
