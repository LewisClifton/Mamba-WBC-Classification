import torch
import torch.nn as nn

torch.backends.cudnn.enabled = True


# Simple ensemble model to improve results


class Ensemble(nn.Module):
    def __init__(self, models, num_classes, device):
        super().__init__()

        self.models = models
        self.models_device = device

        in_features = sum([self._get_model_output_size(model) for model in models])

        self.fc = nn.Linear(in_features, num_classes)

    def _get_model_output_size(self, model):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.models_device)
            base_output = model(dummy_input)
            
            return base_output.shape[1]

    def forward(self, x):
    
        output = self.fc(x)

        return output

