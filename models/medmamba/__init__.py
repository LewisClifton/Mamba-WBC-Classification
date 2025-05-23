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


def __build_model(model_size, num_classes):
    if model_size == 'tiny':
        return MedMamba(depths=[2, 2, 4, 2], dims=[96,192,384,768], num_classes=num_classes)
    if model_size == 'small':
        return MedMamba(depths=[2, 2, 8, 2], dims=[96,192,384,768], num_classes=num_classes)
    if model_size == 'base':
        return MedMamba(depths=[2, 2, 12, 2], dims=[128,256,512,1024], num_classes=num_classes)

def get(num_classes, pretrained_model_path):

    # Load pretrained weights
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
    
        # Load the model
        pretrained_num_classes = state_dict["head.weight"].shape[0]
        model = __build_model('tiny', pretrained_num_classes)
        model.load_state_dict(state_dict, strict=False)
    
    else:
        model = __build_model('tiny', num_classes)

    # Adjust model head
    if num_classes is None:
        # Remove head if necessary
        model.head = nn.Identity()
    else:
        # Change model head
        pretrained_num_classes = model.head.out_features
        if num_classes != pretrained_num_classes:
            model.head = nn.Linear(model.head.in_features, num_classes)

    return model, TRANSFORM_MEDMAMBA