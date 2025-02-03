# Getting started for Linux

- Install [PyTorch for Linux](https://pytorch.org/get-started/locally/)

- ```pip install packaging```
- ```pip install timm==0.4.12```
- ```pip install pytest chardet yacs termcolor```
- ```pip install submitit tensorboardX```
- ```pip install triton==2.2.0```
- ```pip install causal_conv1d```
- ```pip install mamba_ssm```
- ```pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs pillow pandas```



## Usage

Example training config files for the available datasets can be found [here](https://github.com/LewisClifton/LeukaemiaClassification/tree/main/config).

Note on configs: some config files require editing prior to running for example to enter the paths to local datasets. Check config files before running to ensure these paths are set correctly. 

Note on modification of configs: different datasets may require differently structured config files so make note of the structure used in the files provided when making modifications. 

- Train SWIN Transformer using private Chula dataset:
```
python train.py --config_path='config/chula/swin_config' --out_dir='.' --num_gpus=1
```

- Train SWIN Transformer using BloodMNIST dataset:
```
python train.py --config_path='config/bloodmnist/swin_config' --out_dir='.' --num_gpus=1
```