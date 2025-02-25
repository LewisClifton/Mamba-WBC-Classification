import torch.nn as nn
from torchvision import transforms
import torch


TRANSFORM_SWIN = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=25),
        transforms.RandomHorizontalFlip(p=1.0), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


def __build_model():

    model_size = 'tiny'
    if model_size == 'tiny':
        from torchvision.models import swin_t
        return swin_t(weights='IMAGENET1K_V1')

    elif model_size == 'small':
        from torchvision.models import swin_s
        return swin_s(weights='IMAGENET1K_V1')

    elif model_size == 'base': 
        from torchvision.models import swin_b
        return swin_b(weights='IMAGENET1K_V1')


def get(num_classes, pretrained_model_path):

    model = __build_model()

    # Load pretrained weights first
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        # Build the model from the pretrained
        pretrained_num_classes = state_dict["head.weight"].shape[0]
        
        model.head = nn.Linear(model.head.in_features, pretrained_num_classes)
        model.load_state_dict(state_dict, strict=False)
    
    # Change model head
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model, TRANSFORM_SWIN