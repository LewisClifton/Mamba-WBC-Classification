import torch.nn as nn
import torch
import torch.nn.functional as F


class NucleusExtractor(nn.Module):
    def __init__(self):
        super(NucleusExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Learnable feature extractor
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Outputs a "mask"
        self.sigmoid = nn.Sigmoid()  # Converts mask to values between 0 and 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        mask = self.sigmoid(self.conv2(x))  # Predicted mask
        return mask


class Wrapper(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(Wrapper, self).__init__()
        
        self.base_model = base_model

        self.nucleus_extractor = NucleusExtractor()

        self.morph_fc = nn.Sequential(
            nn.Linear(128, 64),  # Morphological features input
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Final classification layer (image features + morph features)
        self.head = nn.Linear(2048 + 128, num_classes)


    def forward(self, image):
        # Extract the nucleus mask
        mask = self.nucleus_extractor(image)  # [B, 1, H, W]

        # Apply the mask to the original image
        nucleus = image * mask  # Keeps only nucleus pixels

        # Extract features from the nucleus
        img_features = self.base_model(nucleus)

        # Learn morphological features (e.g., shape, size, convexity)
        morph_features = self.morph_fc(torch.flatten(nucleus, start_dim=1))

        # Combine image & morphological features
        combined = torch.cat([img_features, morph_features], dim=1)

        return self.head(combined)
    

def wrap_model(base_model, num_classes, pretrained_model_path):
    # Load the WBC Classifier Wrapper
    model = Wrapper(base_model=base_model, num_classes=num_classes)

    # Load pretrained weights if provided
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        pretrained_num_classes = state_dict["head.weight"].shape[0]
        model.head = nn.Linear(model.head.in_features, pretrained_num_classes)
        model.load_state_dict(state_dict, strict=False)

    # Change model head
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model