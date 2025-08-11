import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Student(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)  # or True
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    

def get(num_classes):

    # Build the model
    model = Student(num_classes)

    return model
