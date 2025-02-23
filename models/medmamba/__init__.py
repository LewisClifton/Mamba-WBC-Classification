from torchvision import transforms
import torch
import torch.nn as nn

from .medmamba import VSSM as MedMamba


TRANSFORM_MEDMAMBA = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


def get(num_classes, pretrained_model_path):

    model_size = 'tiny'
    if model_size == 'tiny':
        model = MedMamba(depths=[2, 2, 4, 2], dims=[96,192,384,768], num_classes=6)
    if model_size == 'small':
        model = MedMamba(depths=[2, 2, 8, 2], dims=[96,192,384,768], num_classes=6)
    if model_size == 'base':
        model = MedMamba(depths=[2, 2, 12, 2], dims=[128,256,512,1024], num_classes=6)


    # Load pretrained weights if provided
    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)
    
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model, TRANSFORM_MEDMAMBA