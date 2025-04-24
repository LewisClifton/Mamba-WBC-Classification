#!/bin/bash

#SBATCH --job-name=prepare_dataset
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --account=coms030646


cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

python3 datasets/chula/prepare.py --images_dir="/user/work/js21767/Project/data/WBC 5000/" --labels_dir="/user/work/js21767/Project/data/"

python3 class_weights.py --dataset_config_path="my_configs/datasets/chula_augmented.yml"