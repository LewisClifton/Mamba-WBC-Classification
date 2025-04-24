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
TRAINED_MODEL_PATHS=("/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_localmamba/localmamba_fold_1_acc_92.00437871674211.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_mambavision/mambavision_fold_2_acc_93.50056839396288.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_medmamba/medmamba_fold_4_acc_91.17585927454348.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_swin/swin_fold_3_acc_93.0000834244823.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_41_vim/vim_fold_4_acc_90.56094454657911.pth"
                     "/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_41_vmamba/vmamba_fold_1_acc_93.20851640309058.pth"
                     )

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
TRAINED_MODEL_PATH=${TRAINED_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}

# Debugging output
echo "Evaluating model: $MODEL_NAME"

# Paths
OUT_DIR="/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/eval/"
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
