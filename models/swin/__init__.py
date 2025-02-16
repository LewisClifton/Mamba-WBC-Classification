import torch.nn as nn

def get_swin(num_classes):

    model_size = 'tiny'
    if model_size == 'tiny':
        from torchvision.models import swin_t
        model = swin_t(weights='IMAGENET1K_V1')

    elif model_size == 'small':
        from torchvision.models import swin_s
        model = swin_s(weights='IMAGENET1K_V1')

    elif model_size == 'base': 
        from torchvision.models import swin_b
        model = swin_b(weights='IMAGENET1K_V1')

    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model