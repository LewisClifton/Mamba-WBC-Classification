#!/bin/bash

#SBATCH --job-name=prepare_dataset
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --account=coms030646


cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

python3 datasets/bloodmnist/prepare.py --out_dir="/user/work/js21767/Project/data/BloodMNIST/"

python3 class_weights.py --dataset_config_path="my_configs/datasets/bloodmnist_augmented.yml" --dataset_download_dir="/user/work/js21767/Project"