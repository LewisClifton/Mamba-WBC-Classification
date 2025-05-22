# Mamba-based WBC classification

An application and evaluation of Mamba-based methods for White Blood Cell (WBC) classification. 

Supported models:
- Swin Transformer (for comparison) ([code](https://pytorch.org/vision/main/models/swin_transformer.html), [paper](http://arxiv.org/abs/2103.14030))
- Vim ([code](https://github.com/hustvl/Vim), [paper](https://arxiv.org/abs/2401.09417))
- MambaVision ([code](https://github.com/NVlabs/MambaVision), [paper](https://arxiv.org/abs/2407.08083))
- LocalMamba ([code](https://github.com/hunto/LocalMamba), [paper](https://arxiv.org/abs/2403.09338))
- VMamba ([code](https://github.com/MzeroMiko/VMamba), [paper](https://arxiv.org/abs/2401.10166))
- MedMamba ([code](https://github.com/YubiaoYue/MedMamba), [paper](https://arxiv.org/abs/2403.03849))

Completed as part of my undergraduate dissertation at the University of Bristol.

*Note: This repository is currently a work-in-progress. As a result, the current documentation may be out of date.*

## Supported Datasets
- "Chula-WBC-8" WBC dataset ([data](https://huggingface.co/datasets/LewisClifton/Chula-WBC-8))
- BloodMNIST dataset ([code](https://github.com/MedMNIST/MedMNIST), [paper](https://www.nature.com/articles/s41597-022-01721-8), [data](https://doi.org/10.5281/zenodo.10519652))

## Requirements
- Windows or Linux (multi-GPU support exclusive to Linux)
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

- Then follow the OS Specific instructions:
  
  - For [Windows](https://github.com/LewisClifton/LeukaemiaClassification/blob/main/docs/windows.md) 

  - For [Linux](https://github.com/LewisClifton/LeukaemiaClassification/blob/main/docs/linux.md)


# Results

Coming soon...

# Acknowledgements
Thank you to all of the authors of the work that this project is based on.
