#!/bin/bash

#SBATCH --job-name=ensemble
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --account=coms030646
# #SBATCH --array=0-1  # Adjust based on the number of models
SLURM_ARRAY_TASK_ID=0
DATASET_CONFIG_NAMES=("chula_mamba_meta_learner" "chula_swin_meta_learner")
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}
DATASET_CONFIG_PATH="my_configs/datasets/meta_learner/${DATASET_CONFIG_NAME}.yml"

META_LEARNER_NAMES=("ensemble_mamba" "ensemble_swin")
META_LEARNER_NAME=${META_LEARNER_NAMES[$SLURM_ARRAY_TASK_ID]}
META_LEARNER_CONFIG_PATH="/user/home/js21767/Project/my_configs/models_train/ensemble/${META_LEARNER_NAME}.yml"

TEST_NAME="5epoch_${META_LEARNER_NAME}"
OUT_DIR="/user/work/js21767/Project/out/chula/ensemble/meta_learner/best/${TEST_NAME}/"

DATASET_DOWNLOAD_DIR="/user/work/js21767/"

cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

python3 train.py --out_dir "$OUT_DIR" \
                --model_config_path "$META_LEARNER_CONFIG_PATH" \
                --dataset_config_path "$DATASET_CONFIG_PATH" \
                --num_folds=1 \
                --dataset_download_dir "$DATASET_DOWNLOAD_DIR" \
                --verbose