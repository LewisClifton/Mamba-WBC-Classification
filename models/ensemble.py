import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = True


class Ensemble(nn.Module):
    def __init__(self, base_models, num_classes, device):
        super().__init__()

        in_features = len(base_models) * num_classes

        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, in_features // 4)
        self.fc3 = nn.Linear(in_features // 4, num_classes)

        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(in_features // 2)
        self.bn2 = nn.BatchNorm1d(in_features // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

