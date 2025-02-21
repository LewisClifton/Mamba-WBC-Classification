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

    model = MedMamba(num_classes=num_classes)

    # Load pretrained weights if provided
    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)
    
    model.head = nn.Linear(model.model.head.in_features, num_classes)

    return model, TRANSFORM_MEDMAMBA