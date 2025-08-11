import torch.nn as nn
from torchvision import transforms
import torch


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

    # Load model
    model = __build_model()

    # Load pretrained weights
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        # Build the model from the pretrained
        pretrained_num_classes = state_dict["head.weight"].shape[0]
        
        model.head = nn.Linear(model.head.in_features, pretrained_num_classes)
        model.load_state_dict(state_dict, strict=False)

    if num_classes is None:
        # Remove head if necessary
        model.head = nn.Identity()
    else:
        pretrained_num_classes = model.head.out_features
        # Change model head
        if num_classes != pretrained_num_classes:
            model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model