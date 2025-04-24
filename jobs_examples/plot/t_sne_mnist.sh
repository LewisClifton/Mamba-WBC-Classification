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


# python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/vmamba_fold_4_acc_95.69120287253142.pth" --model_type="vmamba" --dataset_config_path="my_configs/datasets/chula.yml"
python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_swin/swin_fold_3_acc_93.0000834244823.pth" --model_type="swin" --dataset_config_path="my_configs/datasets/chula.yml"
python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/chula/no_pretrain_orig/baseline/2025_04_12_AM10_38_localmamba/localmamba_fold_1_acc_92.00437871674211.pth" --model_type="localmamba" --dataset_config_path="my_configs/datasets/chula.yml"

# python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM12_55_swin/swin_fold_2_acc_99.08026755852842.pth" --model_type="swin" --dataset_config_path="my_configs/datasets/bloodmnist.yml"
# python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/mnist/no_aug/Baseline/2025_04_05_AM12_55_localmamba/localmamba_fold_2_acc_99.1638795986622.pth" --model_type="localmamba" --dataset_config_path="my_configs/datasets/bloodmnist.yml"
 
# python3 t_sne.py --dataset_config_path="my_configs/datasets/chula.yml" --dataset_download_dir="/user/work/js21767/Project/"

# python3 t_sne.py --dataset_config_path="my_configs/datasets/chula.yml" --dataset_download_dir="/user/work/js21767/Project/"

# python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/chula/ensemble/trained_base_models/2025_04_16_AM11_32_vmamba/vmamba_acc_0.pth" --model_type="vmamba" --dataset_config_path="my_configs/datasets/chula.yml"

# python3 t_sne.py --trained_model_path="/user/work/js21767/Project/out/chula/ensemble/trained_base_models/2025_04_16_AM11_31_swin/swin_acc_0.pth" --model_type="swin" --dataset_config_path="my_configs/datasets/chula.yml"