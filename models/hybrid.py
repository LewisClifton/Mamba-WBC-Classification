import numpy as np
import cv2
from PIL import Image
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


TRANSFORM_HYBRID = {    
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


class HybridClassifier(nn.Module):
    def __init__(self, base_model, num_classes, fine_tuning=False):
        super().__init__()

        # Freeze the base model if training, unfreeze if doing the final fine-tuning
        if not fine_tuning:
            base_model.eval()
        else:
            base_model.train()

        self.base_model = base_model
        self.fine_tuning = fine_tuning

        # Feature dimension to project both base model features and morphological features to
        d_model = 24
        self.d_model =24

        # Morphological feature encoder
        self.morph_encoder = nn.Sequential(
            nn.BatchNorm1d(25) # Based on output size of _extract_morph_features
            nn.Linear(25, d_model * 2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model * 2),
            nn.Dropout(0.5),
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.5),
        )

        # Get the output size of the base model features (which varies across architectures)
        base_model_out_features = self._get_model_output_size(self.base_model)

        # Project these features to the specified feature dimension
        self.base_proj = nn.Linear(base_model_out_features, d_model)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # WBC classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes) # Head
        )

    @staticmethod
    def _get_model_output_size(model):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).cuda()
            base_output = model(dummy_input)
            return base_output.shape[1]

    @staticmethod
    def _count_lobes_and_areas(mask):
        """
        Count the number of lobes and compute the area of each for a binary nucleus mask. 

        Args:
            mask (nump.ndarray): Binary nucleus mask

        Returns:
            int: Number of lobes
            list[float]: Area of each lobe above area 100px^2
        """

        num_lobes, pixels = cv2.connectedComponents(image)

        areas = []
        for lobe in range(1, num_lobes):
            area = np.sum(pixels == lobe)

            if area > 100:
                areas.append(area.item())

        return len(areas), areas

    @staticmethod
    def _extract_morph_features(image):
        """
        Extract erosion-based morphological features. 

        Args:
            image (nump.ndarray): Image

        Returns:
            nump.ndarray: Morphological features
        """

        # Convert to HSV colour space
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)

        # Threshold to extract nucleus
        lower_purple = np.array([155, 50, 120])
        upper_purple = np.array([180, 255, 160])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # Do morphological close to fill holes in nucleus mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Get structuring element for the erosion operations
        structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Define erosion iterations
        erosion_iters = [0, 3, 5, 7, 9]
        
        features = []

        # Do the iterative erosion
        for i, it in enumerate(erosion_iters):
            if i == 0:
                eroded = mask # don't do erosion initially
            else:
                eroded = cv2.erode(eroded, structure_element, iterations=erosion_iters[i])
            
            # Get features for the current level of erosion
            num_lobes, areas = HybridClassifier._count_lobes_and_area(eroded)

            # Only consider the four highest love areas
            largest = sorted(areas, reverse=True)[:4] + [0] * (4 - len(areas))
            largest.append(num_lobes)
            features.extend(largest) # 5 features
        
        return np.array(features, dtype=np.float32) # shape (num_erosion_iteration * 5)

    @staticmethod
    def _denormalise(image_tensors):
        """
        Denormalise image tensor based on the imagenet-1k mean and std to allow for image operations. 

        Args:
            image_tensors (torch.tensor): Image batch

        Returns:
            torch.tensor: Denormalised image batch
        """

        denormalise_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]), # imagenet-1k mean
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]), # imagenet-1k std
        ])

        return denormalise_transform(image_tensors) * 255

    def forward(self, x):

        # Don't do backprop for base model unless doing the final fine-tuning step
        if not self.fine_tuning:
            with torch.no_grad():
                base_out = self.base_model(x)
        else:
            base_out = self.base_model(x)

        # Morphological feature pipeline
        x_np = HybridClassifier._denormalise(x).detach().cpu().permute(0, 2, 3, 1).numpy() # denormalise image so can do image processing
        morph_features = np.array([HybridClassifier._extract_morph_features(img) for img in x_np]) # get feautes for each
        morph_features = torch.tensor(morph_features, dtype=torch.float32, device=x.device) # convert features to tensor
        morph_features = self.morph_norm(morph_features) # normalise features
        encoded_morph_features = self.encoder(morph_features) # pass features through the encoder

        # Project the base model output to the same dimensions as the morphological features
        base_proj = self.base_proj(base_out)

        # Gate the base model features and morphological features
        gate_input = torch.cat([base_proj, encoded_morph_features], dim=1)
        gate_weight = self.gate(gate_input)
        # print(f"gate  mean: {gate_weight.mean().item()}") # gives the average weight of features, typically ~0.75 so mostly uses Mamba features

        # Combine features with gate weight
        gated_feat = gate_weight * base_proj + (1 - gate_weight) * encoded_morph_features

        # Final classifier
        out = self.classifier(gated_feat)

        return out


def get(base_model, num_classes, pretrained_model_path):

    if pretrained_model_path is not None:
        model = HybridClassifier(base_model, num_classes, fine_tuning=True)
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)

    else:
        # Build the model
        model = HybridClassifier(base_model, num_classes, fine_tuning=False)

    return model
