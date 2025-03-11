import torch
import torch.nn as nn

torch.backends.cudnn.enabled = True


# Simple ensemble model to improve results


class Ensemble(nn.Module):
    def __init__(self, base_models, num_classes, device):
        super().__init__()

        in_features = self._get_in_features(base_models, device)

        hidden_dim=256
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = nn.Dropout(0.3)


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
    
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

