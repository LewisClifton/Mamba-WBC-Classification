from transformers import AutoModelForImageClassification
import torch.nn as nn
from torchvision import transforms
import torch
        

# Need a wrapper to remove the dict wrapping of mambavision.forward
class MambaVisionWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the pretrained model
        model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True).model

        self.model = model


    def forward(self, x):
        return self.model(x)


def get(num_classes, pretrained_model_path):
    
    # Load the model
    model = MambaVisionWrapper()
    pretrained_num_classes = model.model.head.out_features

    # Load pretrained weights
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        # Build the model from the pretrained
        pretrained_num_classes = state_dict["model.head.weight"].shape[0]
        
        model.model.head = nn.Linear(model.model.head.in_features, pretrained_num_classes)
        model.load_state_dict(state_dict, strict=False)

    # Adjust model head
    if num_classes is None:
        # Remove head if necessary
        model.model.head = nn.Identity()
    else:
        # Change model head
        if num_classes != pretrained_num_classes:
            model.model.head = nn.Linear(model.model.head.in_features, num_classes)

    return model