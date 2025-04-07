import numpy as np
import cv2
from PIL import Image
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


TRANSFORM_NEUTROPHILS = {    
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
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


# Two-tier model making use of a bespoke neutrophils classifier to try to improve results
class CompleteClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()

        self.base_model = base_model
        self.base_model.eval()
        base_model_out_features = self._get_model_output_size(self.base_model)

        self.encoder = nn.Sequential(
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(0.5),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.Dropout(0.5),
        )

        self.base_proj = nn.Linear(base_model_out_features, 128)
        self.morph_proj = nn.Linear(24, 128)
        self.gate = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )


    @staticmethod
    def _get_model_output_size(model):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).cuda()
            base_output = model(dummy_input)
            return base_output.shape[1]

    @staticmethod
    def _count_islands_and_area(image):
        num_islands, pixels = cv2.connectedComponents(image)
        areas = []
        for island in range(1, num_islands):
            area = np.sum(pixels == island)
            if area > 100:
                areas.append(area.item())
        return len(areas), areas

    @staticmethod
    def _extract_morph_features(image):
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
        lower_purple = np.array([155, 50, 120])
        upper_purple = np.array([180, 255, 160])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion_iters = [0, 3, 5, 7, 9, 11]

        features = []
        for i, it in enumerate(erosion_iters):
            if i == 0:
                eroded = mask
            else:
                eroded = cv2.erode(eroded, kernel, iterations=erosion_iters[i])
            _, areas = CompleteClassifier._count_islands_and_area(eroded)
            largest = sorted(areas, reverse=True)[:4] + [0] * (4 - len(areas))
            features.extend(largest)
        return np.array(features, dtype=np.float32)

    @staticmethod
    def denormalize(image_tensor):
        invTrans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])
        return invTrans(image_tensor) * 255

    def forward(self, x):
        with torch.no_grad():
            base_out = self.base_model(x)

        # Morph feature processing
        x_np = CompleteClassifier.denormalize(x).detach().cpu().permute(0, 2, 3, 1).numpy()
        morph_features = np.array([CompleteClassifier._extract_morph_features(img) for img in x_np])
        morph_features = torch.tensor(morph_features, dtype=torch.float32, device=x.device)

        # Project both to 128-dim space
        base_proj = self.base_proj(base_out)
        morph_proj = self.morph_proj(morph_features)

        # Gating
        gate_input = torch.cat([base_proj, morph_proj], dim=1)
        gate_weight = self.gate(gate_input)

        # Combine via gate
        gated_feat = gate_weight * morph_proj + (1 - gate_weight) * base_proj

        # Final classifier
        out = self.classifier(gated_feat)

        return out




def get(base_model, num_classes, pretrained_model_path):

    # Build the model
    model = CompleteClassifier(base_model, num_classes)

    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)
    
    return model
