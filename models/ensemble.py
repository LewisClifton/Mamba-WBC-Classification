import torch
import torch.nn as nn

torch.backends.cudnn.enabled = True


# Simple ensemble model to improve results


class Ensemble(nn.Module):
    def __init__(self, base_models, num_classes, device):
        super().__init__()

        in_features = self._get_in_features(base_models, device)

        self.fc = nn.Linear(in_features, num_classes)


    def _get_in_features(self, base_models, device):
        
        in_features = 0

        for base_model in base_models:
            base_model = base_model.to(device)

            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                base_output = base_model(dummy_input)
                
                in_features += base_output.shape[1]

            base_model.to('cpu')

        return in_features


    def forward(self, x):
    
        output = self.fc(x)

        return output

