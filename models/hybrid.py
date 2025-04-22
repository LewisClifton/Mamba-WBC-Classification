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

        # Get the output size of the base model features (which varies across architectures)
        base_model_out_features = self._get_model_output_size(self.base_model)

        # Project these features to the specified feature dimension
        self.base_proj = nn.Sequential(
            #nn.BatchNorm1d(base_model_out_features),
            nn.Linear(base_model_out_features, base_model_out_features // 2),
            nn.Linear(base_model_out_features // 2, d_model),
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
            nn.Linear(d_model // 4, num_classes) # Head
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, num_classes),
        #     #nn.ReLU(),
        #     #nn.Linear(16, num_classes) # Head
        # )

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
            num_lobes, areas = HybridClassifier._count_lobes_and_areas(eroded)

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
        mask = HybridClassifier._get_mask(image)
        
        # Get 36 mask features
        feature_names = ['axis_major_length', 'perimeter', 'solidity', 'eccentricity', 'moments_hu', 'area']
        features = HybridClassifier._get_mask_features(mask, feature_names)

        # Get a further 20 erosion features
        erosion_iters = [3, 5, 7, 9]
        erosion_features = HybridClassifier._get_erosion_features(mask, erosion_iters)
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

    def forward(self, x):

        # Don't do backprop for base model unless doing the final fine-tuning step
        if not self.fine_tuning:
            with torch.no_grad():
                base_out = self.base_model(x)
        else:
            base_out = self.base_model(x)

        base_pred = torch.argmax(wbc_out, dim=1)

         # Mask for images classified as neutrophils
        neutrophil_mask = torch.isin(wbc_type, self.neutrophils_indices.to(wbc_type.device))

        if neutrophil_mask.any():
            # Get indices of images classified as neutrophils
            neutrophil_indices = torch.where(neutrophil_mask)[0]

            # Get neutrophil type prediction
            neutrophil_out = self.neutrophils_model(x[neutrophil_indices])
            neutrophil_type = torch.argmax(neutrophil_out, dim=1)

            # Map the neutrophils binary predictions to WBC classes
            wbc_type[neutrophil_indices] = torch.where(neutrophil_type == 1, self.BNE_index, self.SNE_index)
        

        # Morphological feature pipeline
        x_np = HybridClassifier._denormalise(x).detach().cpu().permute(0, 2, 3, 1).numpy() # denormalise image so can do image processing
        morph_features = np.array([HybridClassifier._extract_morph_features(img) for img in x_np]) # get feautes for each
        morph_features = torch.tensor(morph_features, dtype=torch.float32, device=x.device) # convert features to tensor
        encoded_morph_features = self.morph_encoder(morph_features) # pass features through the encoder

        # Project the base model output to the same dimensions as the morphological features
        base_proj = self.base_proj(base_out)

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
    def __init__(self, base_model, neut_model, num_classes, fine_tuning=False):
        super().__init__()

        # Freeze the base model if training, unfreeze if doing the final fine-tuning
        if not fine_tuning:
            base_model.eval()
        else:
            base_model.train()

        self.base_model = base_model
        self.fine_tuning = fine_tuning
        self.neut_model = 

    @staticmethod
    def _get_model_output_size(model):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).cuda()
            base_output = model(dummy_input)
            return base_output.shape[1]



    def forward(self, x):

        # Don't do backprop for base model unless doing the final fine-tuning step
        if not self.fine_tuning:
            with torch.no_grad():
                base_out = self.base_model(x)
        else:
            base_out = self.base_model(x)

        base_pred = torch.argmax(wbc_out, dim=1)

         # Mask for images classified as neutrophils
        neutrophil_mask = torch.isin(wbc_type, self.neutrophils_indices.to(wbc_type.device))

        if neutrophil_mask.any():
            # Get indices of images classified as neutrophils
            neutrophil_indices = torch.where(neutrophil_mask)[0]

            # Get neutrophil type prediction
            neutrophil_out = self.neutrophils_model(x[neutrophil_indices])
            neutrophil_type = torch.argmax(neutrophil_out, dim=1)

            # Map the neutrophils binary predictions to WBC classes
            wbc_type[neutrophil_indices] = torch.where(neutrophil_type == 1, self.BNE_index, self.SNE_index)
        

        return out



def get(base_model, num_classes, pretrained_model_path):

    if pretrained_model_path is not None:
        model = HybridClassifier(base_model, num_classes, fine_tuning=True)
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)

    else:
        # Build the model
        model = HybridClassifier(base_model, num_classes, fine_tuning=False)

    return model
