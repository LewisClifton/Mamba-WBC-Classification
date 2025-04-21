import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = True

class MetaLearner(nn.Module):
    def __init__(self, num_base_models, num_classes):
        super().__init__()

        in_features = num_base_models * num_classes

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


def get(num_base_models, num_classes, pretrained_model_path):

    # Build the model
    model = MetaLearner(num_base_models, num_classes)

    # Load from pretrained
    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)
    
    return model, None
