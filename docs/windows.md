# Getting started for Windows

- Install [PyTorch for Windows](https://pytorch.org/get-started/locally/)
- ```pip install pillow sklearn pandas```
- ```pip install medmnist```

## Usage

Example training config files for the available datasets can be found [here](https://github.com/LewisClifton/LeukaemiaClassification/tree/main/config).

Note on configs: some config files require editing prior to running for example to enter the paths to local datasets. Check config files before running to ensure these paths are set correctly. 

Note on modification of configs: different datasets may require differently structured config files so make note of the structure used in the files provided when making modifications. 

Train SWIN Transformer using private Chula dataset:
```
python train.py --config_path='config/chula/swin_config' --out_dir='.' --num_gpus=1
```

Train SWIN Transformer using BloodMNIST dataset:
```
python train.py --config_path='config/bloodmnist/swin_config' --out_dir='.' --num_gpus=1
```