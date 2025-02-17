from transformers import AutoModelForImageClassification
import torch.nn as nn
        
# Need a wrapper to remove the dict wrapping of mambavision.forward
class MambaVisionWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


def get_mambavision(num_classes):
    model = MambaVisionWrapper(num_classes=num_classes)

    # To do: change heda

    return model