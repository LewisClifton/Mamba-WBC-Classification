# Getting started for Windows

- Install [PyTorch for Windows](https://pytorch.org/get-started/locally/)
- ```pip install pillow sklearn pandas```
- ```pip install medmnist```

## Usage

Example training configuration files for the available datasets can be found [here](https://github.com/LewisClifton/LeukaemiaClassification/tree/main/config).

Note: different datasets may require differently structured config files  so make note of the structure used in the files provided when making modifications.

Train SWIN Transformer using private Chula dataset:
```
python train.py --config_path='config/chula/swin_config' --out_dir='.' --num_gpus=1
```

Train SWIN Transformer using BloodMNIST dataset:
```
python train.py --config_path='config/bloodmnist/swin_config' --out_dir='.' --num_gpus=1
```