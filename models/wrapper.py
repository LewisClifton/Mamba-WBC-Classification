import torch.nn as nn
import torch
import torch.nn.functional as F


# A wrapper for models to try and improve results

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

        self.base_model_device = next(self.base_model.parameters()).device
        self.nucleus_extractor = NucleusExtractor()

        # CNN for learning morphological features from the extracted nucleus
        self.morph_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial size
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: [B, 64, 1, 1]
        )

        self.morph_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # **Dynamically determine base model output size**
        self.base_out_features = self._get_base_output_size()

        # Final classifier layer (base model features + morphological features)
        self.head = nn.Linear(self.base_out_features + 128, num_classes)

    def _get_base_output_size(self):
        """Pass a dummy input to determine base model output size."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.base_model_device)
            base_output = self.base_model(dummy_input)
            
            return base_output.shape[1]  # Get feature size

    def forward(self, image):
        # Extract nucleus mask
        mask = self.nucleus_extractor(image)  # [B, 1, H, W]

        # Apply mask to extract nucleus
        nucleus = image * mask  # [B, 3, H, W]

        # Extract features from the nucleus image (Base Model)
        img_features = self.base_model(nucleus)  # [B, 192]
        
        # Pass nucleus through CNN for morphological feature extraction
        morph_features = self.morph_cnn(nucleus)  # [B, 64, 1, 1]
        morph_features = morph_features.view(morph_features.size(0), -1)  # Flatten to [B, 64]

        # Pass through FC layers
        morph_features = self.morph_fc(morph_features)  # [B, 128]

        # Concatenate features
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