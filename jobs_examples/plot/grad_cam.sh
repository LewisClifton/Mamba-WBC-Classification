#!/bin/bash

#SBATCH --job-name=t_sne
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --account=coms030646

cd /user/home/js21767/Project/
source ~/miniconda3/bin/activate
conda activate leuk_classification

python3 grad_cam.py --trained_model_path="/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_swin/swin_fold_3_acc_93.0000834244823.pth" --model_type="swin" --num_classes=8
python3 grad_cam.py --trained_model_path="/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_localmamba/localmamba_fold_1_acc_92.00437871674211.pth" --model_type="localmamba" --num_classes=8