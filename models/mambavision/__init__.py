from transformers import AutoModelForImageClassification
import torch.nn as nn
from torchvision import transforms
        
        
TRANSFORM_MAMBAVISION = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# Need a wrapper to remove the dict wrapping of mambavision.forward
class MambaVisionWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)
        model.model.head = nn.Linear(model.model.head.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


def get(num_classes):
    model = MambaVisionWrapper(num_classes=num_classes)

    # To do: change heda

    return model, TRANSFORM_MAMBAVISION