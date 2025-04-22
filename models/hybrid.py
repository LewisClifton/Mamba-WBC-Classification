import numpy as np
import cv2
from PIL import Image
import math
from skimage.measure import label, regionprops_table
import pandas as pd

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


class NeutClassifier(nn.Module):
    def __init__(self, base_model_output_size):
        super().__init__()

        # Feature dimension to project both base model features and morphological features to
        d_model = 256
        self.d_model = d_model

        # Morphological feature encoder
        self.morph_encoder = nn.Sequential(
            nn.BatchNorm1d(56), # Based on output size of _extract_morph_features
            nn.Linear(56, d_model // 2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(d_model),
            #nn.Dropout(0.5),
        )

        # Project these features to the specified feature dimension
        self.base_proj = nn.Sequential(
            #nn.BatchNorm1d(base_model_output_size),
            nn.Linear(base_model_output_size, base_model_output_size // 2),
            nn.Linear(base_model_output_size // 2, d_model),
            nn.BatchNorm1d(d_model),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.BatchNorm1d(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # WBC classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1) # Binary classifying head
        )


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

        num_lobes, pixels = cv2.connectedComponents(mask)

        areas = []
        for lobe in range(1, num_lobes):
            area = np.sum(pixels == lobe)

            if area > 100:
                areas.append(area.item())

        return len(areas), areas

    @staticmethod
    def _get_mask(image):
        # Convert to HSV colour space
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)

        # Threshold to extract nucleus
        lower_purple = np.array([155, 50, 120])
        upper_purple = np.array([180, 255, 160])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # Do morphological close to fill holes in nucleus mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    @staticmethod
    def _get_erosion_features(mask, erosion_iters):

        eroded_masks = [mask]
        features = []

        # Get structuring element for the erosion
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Do the iterative erosion
        for i in range(len(erosion_iters)):

            # Erosion
            eroded = cv2.erode(eroded_masks[i], structuring_element, iterations=erosion_iters[i]-(erosion_iters[i-1] or 0)) # erode the previous mask further
            eroded_masks.append(eroded)

            # Get features for the current level of erosion
            num_lobes, areas = NeutClassifier._count_lobes_and_areas(eroded)

            # Only consider the four highest lobe areas
            largest = sorted(areas, reverse=True)[:4] + [0] * (4 - len(areas))
            largest.append(num_lobes)

            # 5 features per erosion level
            features.extend(largest)
        
        return features

    @staticmethod
    def _get_mask_features(mask, feature_names):

        # Extract required features
        labelled_mask = label(mask)
        features = regionprops_table(labelled_mask, properties=feature_names)
        
        # Pad to make three rows
        features = pd.DataFrame(features).reindex(range(3), fill_value=0)
        
        # Get three largest rows
        largest = np.argpartition(features['area'], -3)[:3][::-1] # pad if less than three
        features = features.iloc[largest]

        # Return flat list (length 36)
        return np.array(features).reshape(-1).tolist()
    

    @staticmethod
    def _extract_morph_features(image):
        """
        Extract erosion-based morphological features. 

        Args:
            image (nump.ndarray): Image

        Returns:
            nump.ndarray: Morphological features
        """

        # Get the nucleus mask
        mask = NeutClassifier._get_mask(image)
        
        # Get 36 mask features
        feature_names = ['axis_major_length', 'perimeter', 'solidity', 'eccentricity', 'moments_hu', 'area']
        features = NeutClassifier._get_mask_features(mask, feature_names)

        # Get a further 20 erosion features
        erosion_iters = [3, 5, 7, 9]
        erosion_features = NeutClassifier._get_erosion_features(mask, erosion_iters)
        features.extend(erosion_features)

        # Total output featues = 36 + 20 = 56
        return np.array(features, dtype=np.float32)

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

    def forward(self, x, base_model_features):
        
        # Morphological feature pipeline
        x_np = NeutClassifier._denormalise(x).detach().clone().cpu().permute(0, 2, 3, 1).numpy() # denormalise image so can do image processing
        morph_features = np.array([NeutClassifier._extract_morph_features(img) for img in x_np]) # get feautes for each
        morph_features = torch.tensor(morph_features, dtype=torch.float32, device=x.device) # convert features to tensor
        morph_features = morph_features.to('cuda:0')
        encoded_morph_features = self.morph_encoder(morph_features) # pass features through the encoder

        # Project the base model output to the same dimensions as the morphological features
        base_proj = self.base_proj(base_model_features)

        # Gate the base model features and morphological features
        gate_input = torch.cat([base_proj, encoded_morph_features], dim=1)
        gate_weight = self.gate(gate_input)
        
        # print(f"gate  mean: {gate_weight.mean().item()}") # gives the average weight of features, typically ~0.85 so mostly uses Mamba features

        # Combine features with gate weight
        gated_feat = gate_weight * base_proj + (1 - gate_weight) * encoded_morph_features

        # Final classifier
        out = self.classifier(gated_feat)

        return out


class HybridClassifier(nn.Module):
    def __init__(self, num_classes, base_model, fine_tuning=False):
        super().__init__()

        self.fine_tuning = fine_tuning
        
        # Set the forward function based on if not training the neutrophils model and not fine-tuning the hybrid network
        if num_classes == 2:
            self.forward = self.train_forward
        else:
            self.forward = self.test_forward
        
        # If pretrained_base_model is given, implies fine-tuning the whole mode further
        self.fine_tuning = fine_tuning
        self.fine_tuning = True if fine_tuning else False

        if fine_tuning:
            base_model.train()
        else:
            base_model.eval()

        self.base_model = base_model

        # Initialise neutrophils model
        base_model_output_size = self._get_base_model_output_size()
        self.neut_model = NeutClassifier(base_model_output_size)

        # Neutrophils to be update with by NeutClassifier predictions
        self.SNE_index = 0
        self.BNE_index = 2

    def _get_base_model_output_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to('cuda:0')
            base_model_output = self.base_model(dummy_input)
            return base_model_output.shape[1]

    def train_forward(self, x):
        # Train the base model to take combine base model outputs with morphological features

        # Don't do backprop for base model unless doing the final fine-tuning step
        if not self.fine_tuning:
            with torch.no_grad():
                base_out = self.base_model(x)
        else:
            base_out = self.base_model(x)

        neut_type = self.neut_model(x, base_out)

        return neut_type.squeeze(-1)


    def test_forward(self, x):
        # Forward function that uses conditional neutrophils path

        # Get base model output
        base_out = self.base_model(x)
        
        # Get base model prediction
        base_pred = torch.argmax(wbc_out, dim=1)

        # Get images classified as neutrophils
        neutrophil_mask = torch.isin(base_pred, self.neutrophils_indices.to(base_pred.device))

        # if no neutrophils predictions return just return the base model output
        if not neutrophil_mask.any():
            return base_pred 

        # Get indices of images classified as neutrophils
        neutrophil_indices = torch.where(neutrophil_mask)[0]

        # Get confirmed neutrophil type prediction
        neutrophil_out = self.neutrophils_model(x[neutrophil_indices], base_out)
        neutrophil_type = torch.argmax(neutrophil_out, dim=1)

        # Map the neutrophils binary predictions to WBC classes
        base_pred[neutrophil_indices] = torch.where(neutrophil_type == 1, self.BNE_index, self.SNE_index)
        return base_pred


def get(pretrained_model_path, num_classes, base_model):

    
    # Load trained model if provided
    if pretrained_model_path is not None:

        if num_classes == 2:
            # Fine-tuning
            model = HybridClassifier(num_classes=num_classes, base_model=base_model, fine_tuning=True)
        else:
            # Evaluation
            model = HybridClassifier(num_classes=num_classes, base_model=base_model, fine_tuning=False)
            
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)

        model = model.to('cuda:0')
        return model

    else:
        # Initialise fresh model
        model = HybridClassifier(num_classes=num_classes, base_model=base_model).to('cuda:0')

    return model
