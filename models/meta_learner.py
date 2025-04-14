import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = True


class MetaLearner(nn.Module):
    def __init__(self, num_base_models, num_classes):
        super().__init__()

        in_features = num_base_models * num_classes

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


def get(num_base_models, num_classes, pretrained_model_path):

    # Build the model
    model = MetaLearner(num_base_models, num_classes)

    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)
    
    return model, None
