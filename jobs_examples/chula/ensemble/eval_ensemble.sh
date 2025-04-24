#!/bin/bash

#SBATCH --job-name=eval_model_array
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --account=coms030646
# #SBATCH --array=0-1  # Adjust based on the number of models
SLURM_ARRAY_TASK_ID=0
ENSEMBLE_CONFIG_NAMES=("ensemble_mamba" "ensemble_swin")
ENSEMBLE_CONFIG_NAME=${ENSEMBLE_CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}
ENSEMBLE_CONFIG_PATH="/user/home/js21767/Project/my_configs/models_train/ensemble/${ENSEMBLE_CONFIG_NAME}.yml"

META_LEARNER_DIRS=("FIVE_BEST_ensemble_mamba/2025_04_17_PM04_33_meta_learner"            
                   "FIVE_BEST_ensemble_swin/2025_04_17_PM04_33_meta_learner")
META_LEARNER_DIR="/user/work/js21767/Project/out/chula/ensemble/meta_learner/exhaustive/${META_LEARNER_DIRS[$SLURM_ARRAY_TASK_ID]}"
META_LEARNER_PATH="$META_LEARNER_DIR/meta_learner_acc_0.pth"

OUT_DIR="$META_LEARNER_DIR"
DATASET_CONFIG_PATH="my_configs/datasets/chula.yml"
DATASET_DOWNLOAD_DIR="/user/work/js21767/"

cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

python3 eval_ensemble.py --out_dir "$OUT_DIR" \
                --ensemble_config_path "$ENSEMBLE_CONFIG_PATH" \
                --dataset_config_path "$DATASET_CONFIG_PATH" \
                --dataset_download_dir "$DATASET_DOWNLOAD_DIR" \
                --meta_learner_path "$META_LEARNER_PATH" \

