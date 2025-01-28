import torch.nn as nn
from torchvision.models import swin_t, swin_s, swin_b

class SwinWBC(nn.Module):
    def __init__(self, size, ):
        
        # Create model
        if size == 'tiny':
            model = swin_t(weights="IMAGENET1K_V1")
        elif size == 'small':
            model = swin_s(weights="IMAGENET1K_V1")
        else:
            model = swin_b(weights="IMAGENET1K_V1")
        model.head = nn.Linear(model.head.in_features, 8)
        self.model = model

    def forward(self, x):
        
        out = self.model(x)

        return out

