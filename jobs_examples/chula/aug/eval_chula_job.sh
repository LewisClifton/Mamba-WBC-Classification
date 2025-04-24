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
TRAINED_MODEL_PATHS=("/user/work/js21767/Project/out/chula/no_pretrain_aug/baseline/2025_04_21_PM01_58_localmamba/localmamba_fold_3_acc_95.31063370633217.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_aug/baseline/2025_04_21_PM01_58_mambavision/mambavision_fold_4_acc_97.19742391770416.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_aug/baseline/2025_04_21_PM01_58_medmamba/medmamba_fold_1_acc_92.57664908110966.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_aug/baseline/2025_04_21_PM01_58_swin/swin_fold_4_acc_96.14505379678863.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_aug/baseline/2025_04_21_PM03_47_vim/vim_fold_4_acc_95.9489595473775.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_aug/baseline/2025_04_21_PM03_54_vmamba/vmamba_fold_2_acc_96.20046387722108.pth"
                    )

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
TRAINED_MODEL_PATH=${TRAINED_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

# Debugging output
echo "Evaluating model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/chula/no_pretrain_aug/eval/"
DATASET_CONFIG_PATH="my_configs/datasets/chula.yml"
MODEL_CONFIG_PATH="my_configs/models_train/$MODEL_NAME.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"

# Load environment
cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

# Run evaluation script
python3 eval.py --trained_model_path="$TRAINED_MODEL_PATH" \
                --model_config_path="$MODEL_CONFIG_PATH" \
                --dataset_config_path="$DATASET_CONFIG_PATH" \
                --dataset_download_dir="$DATASET_DOWNLOAD_DIR"
