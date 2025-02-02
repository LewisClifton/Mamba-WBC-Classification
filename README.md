# WBC classification using SWIN Transformer and MedMamba



## Requirements
- Windows or Linux
- CUDA drivers
- 1+ Nvidia GPU (multi-GPU support exclusive to Linux)

## Getting started

- Clone this repo:
```
git clone https://github.com/LewisClifton/LeukaemiaClassification.git
cd LeukaemiaClassification
```

- Create conda virtual environment and activate it:
```
conda create -n leukclassification
conda activate leukclassification
```

- OS Specific instructions:
  
  - For [Windows](https://github.com/LewisClifton/LeukaemiaClassification/blob/main/docs/windows.md) 

  - For [Linux](https://github.com/LewisClifton/LeukaemiaClassification/blob/main/docs/linux.md)


## Supported models
- [SWIN Transformer](https://pytorch.org/vision/main/models/swin_transformer.html)

- [MedMamba](https://github.com/YubiaoYue/MedMamba) (Linux only)

## Supported Datasets
- Private "Chula 5000" WBC dataset
- BloodMNIST dataset
