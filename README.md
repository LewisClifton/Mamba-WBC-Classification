<div align="center">

# 🩸🐍 Mamba-based WBC classification 🐍🩸

[![arXiv:2504.11438](https://img.shields.io/badge/arXiv-2504.11438-b31b1b.svg)](https://arxiv.org/abs/2504.11438)
[![Hugging Face](https://img.shields.io/badge/Hugging--Face-Chula--WBC--8-yellow)](https://huggingface.co/datasets/LewisClifton/Chula-WBC-8)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

An application of several Vision Mamba models for White Blood Cell (WBC) classification, undertaken for my Computer Science MEng Final Project at the University of Bristol.

This repository contains:
- `train.py` for the training of standalone models and model ensembles.
- `eval.py` for the evaluation of standalone models.
- `eval_ensemble.py` for the evaluation of model ensembles.
- `grad_cam.py` and `t_sne.py` for the creation of Grad-CAM heatmaps and t-SNE plots, respectively.
- `datasets/` containing an API for the two datasets.
- `configs/*.yml` config files passed as command line arguments for the above scripts.

*Status: Work in progress, results coming soon.*

## Supported models:
- Swin Transformer (for comparison) ([code](https://pytorch.org/vision/main/models/swin_transformer.html), [paper](http://arxiv.org/abs/2103.14030))
- Vim ([code](https://github.com/hustvl/Vim), [paper](https://arxiv.org/abs/2401.09417))
- MambaVision ([code](https://github.com/NVlabs/MambaVision), [paper](https://arxiv.org/abs/2407.08083))
- LocalMamba ([code](https://github.com/hunto/LocalMamba), [paper](https://arxiv.org/abs/2403.09338))
- VMamba ([code](https://github.com/MzeroMiko/VMamba), [paper](https://arxiv.org/abs/2401.10166))
- MedMamba ([code](https://github.com/YubiaoYue/MedMamba), [paper](https://arxiv.org/abs/2403.03849

## Supported Datasets
- Chula-WBC-8" WBC dataset ([data](https://huggingface.co/datasets/LewisClifton/Chula-WBC-8))
- BloodMNIST blood cell dataset ([code](https://github.com/MedMNIST/MedMNIST), [paper](https://www.nature.com/articles/s41597-022-01721-8), [data](https://doi.org/10.5281/zenodo.10519652))

## Requirements
- Linux
- CUDA drivers
- 1+ Nvidia GPU (Turing architecture - Mamba implementation does not support older Pascal architectures)

## Getting started

- Clone this repo:
```
git clone https://github.com/LewisClifton/Mamba-WBC-Classification.git
cd Mamba-WBC-Classification
```

- Create conda virtual environment and activate it:
```
conda create -n mamba_wbc_classification
conda activate mamba_wbc_classification
```

- [Install PyTorch](https://pytorch.org/get-started/locally/)

- Install `environment.yml`:
`conda env update --file environment.yml`

*Note: If unsuccessful, please trying the Manual Installation steps below.*

<details>
<summary>Manual Installation</summary>
<br>
  
- Install [PyTorch for Linux](https://pytorch.org/get-started/locally/)
- Install packages from Bash terminal:
    ```bash
    pip install packaging
    pip install timm==0.4.12
    pip install pytest chardet yacs termcolor
    pip install submitit tensorboardX
    pip install triton==2.2.0
    pip install causal_conv1d
    pip install mamba_ssm
    pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs pillow pandas
    conda install python=3.12
    conda install matplotlib h5py SimpleITK scikit-image medpy yacs pillow anaconda::scikit-learn pandas
    conda install -c conda-forge gcc
    pip3 install torch torchvision torchaudio
    pip3 install mambavision
    pip3 install thop
    git clone https://github.com/MzeroMiko/VMamba.git
    cd VMamba
    pip3 install -r requirements.txt
    pip3 install git+https://github.com/Dao-AILab/causal-conv1d
    conda install gcc_linux-64 gxx-linux-64 -y
    conda install cuda -c nvidia
    git clone https://github.com/state-spaces/mamba.git && cd mamba
    CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE pip install .
    ```

*Note: Please create an Issue if even manual installation fails.*
</details>


# Results

Coming soon...

# Acknowledgements
Thank you to all of the authors of the work that this project is based on.
