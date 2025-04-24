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


class NeutClassifier(nn.Module):
    def __init__(self, d_model, fine_tuning=False):
        super().__init__()

        # Morphological feature encoder
        self.morph_encoder = nn.Sequential(
            nn.BatchNorm1d(61), # Based on output size of _extract_morph_features
            nn.Linear(61, d_model // 4),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 4),
            nn.Dropout(0.5),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(d_model),
        )


    @staticmethod
    def _count_lobes_and_areas(mask):
        """
        Count the number of lobes and compute the area of each lobe segment for a binary nucleus mask. 

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
        """
        Get mask of the nucleus using colour-based thresholding

        Convert the image to HSV colour space
        Extract the deep purple colour nucleus
        Do a morphological close to fill holes in the nucleus

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
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    @staticmethod
    def _get_erosion_features(mask, erosion_iters):
        """
        Extract erosion-based morphological features. 

        Iteratively erode the nucleus mask to break the nucleus into its lobe segments
        Calculate the number of lobe segments and their areas for each level of erosion

        Args:
            image (nump.ndarray): Image

        Returns:
            nump.ndarray: Morphological features
        """

        eroded_masks = [mask]
        features = []

        # Get structuring element for the erosion
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Do the iterative erosion
        for i in range(len(erosion_iters)):

            # Erosion
            prev_erosion_iter = erosion_iters[i-1] if i != 0 else 0
            eroded = cv2.erode(eroded_masks[i], structuring_element, iterations=erosion_iters[i]-(prev_erosion_iter)) # erode the previous mask further
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
        """
        Extract shape features. 

        Args:
            image (nump.ndarray): Image

        Returns:
            nump.ndarray: Morphological features
        """

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
        Extract morphological features. 

        First, get a mask of the nucleus
        Then, get shape features of the mask
        Then obtain erosion-based nucleus segment features

        Return all of these

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
        erosion_iters = [3, 5, 7, 9, 11] # 3
        erosion_features = NeutClassifier._get_erosion_features(mask, erosion_iters)
        features.extend(erosion_features)

        # Total output featues = 36 + 15 = 51
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
        """
        Denormalise image tensor based on the imagenet-1k mean and std to allow for pixel-level image operations. 

        Args:
            x (torch.tensor): Image batch

        Returns:
            torch.tensor: Morphological features
        """
        
        # Morphological feature pipeline
        x_np = NeutClassifier._denormalise(x).detach().cpu().permute(0, 2, 3, 1).numpy() # denormalise image so can do image processing
        morph_features = np.array([NeutClassifier._extract_morph_features(img) for img in x_np]) # get feautes for each
        morph_features = torch.tensor(morph_features, device=x.device) # convert features to tensor

        return self.morph_encoder(morph_features.squeeze(-1)) # pass features through the encoder


class HybridClassifier(nn.Module):
    def __init__(self, num_classes, base_model, fine_tuning=False):
        super().__init__()

        # Feature dimension to project both base model features and morphological features to
        d_model = 512
        self.d_model = d_model
        
        # Set the forward function based on if not training the s model and not fine-tuning the hybrid network
        if num_classes == 2:
            self.forward = self.train_forward
        else:
            self.forward = self.test_forward
        
        # If pretrained_base_model is given, implies fine-tuning the whole mode further
        self.fine_tuning = fine_tuning

        base_model.eval()

        self.base_model = base_model

        # Create a hook that gets the base model feature at the penultimate networ layer
        self.base_model_features = None

        # Make a hook at the final layer to get it's input in the forward pass which will be the Mamba features
        if hasattr(base_model, 'head'):
            self.base_model.head.register_forward_hook(self.base_model_hook_func)
        if hasattr(base_model, 'classifier'):
            if hasattr(base_model.classifier, 'head'):
                self.base_model.classifier.head.register_forward_hook(self.base_model_hook_func)
            else:
                self.base_model.classifier.register_forward_hook(self.base_model_hook_func)
        

        # Initialise neutrophils model
        self.neut_model = NeutClassifier(d_model=d_model, fine_tuning=fine_tuning)

        if fine_tuning:
            self.neut_model.eval()
        else:
            self.neut_model.train()

        # s to be update with by NeutClassifier predictions
        self.SNE_index = 0
        self.BNE_index = 3
        self.neut_indices = torch.tensor([self.SNE_index, self.BNE_index], dtype=int).to(device='cuda:0')

        # Project base model features to the specified feature dimension
        base_model_features_size = self._get_base_model_features_size() # different base models have different size features
        self.base_proj = nn.Sequential(
            #nn.BatchNorm1d(base_model_output_size),
            nn.Linear(base_model_features_size, d_model), 
            nn.BatchNorm1d(d_model),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            #nn.BatchNorm1d(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Neutrophils classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 4, 1) # Binary classifying head
        )

    def base_model_hook_func(self, module, input_, output):
        """
        Return the input features for a given layer of a model during its forward pass. 
        
        Typically used in the HybridClassiforward passfier to obtain the Mamba features from the penultimate layer

        These features are saved to the base_model_features attributes

        Args:
            module (nn.Module): Network layer
            input_ (torch.Tensor): Layer input in the forward pass
            output_ (torch.Tensor): Layer output in the forward pass
        """

        # Hook for extracting base model outputs
        self.base_model_features = input_[0].detach().clone()

    def _get_base_model_features_size(self):
        """
        Get the size of the features for a given base model.

        Necessary as the feature dimension will vary between different base model archtitectures

        Returns:
            tuple: Shape of base model features tensor
        """

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to('cuda:0')
            _ = self.base_model(dummy_input)
            return self.base_model_features.shape[1]

    def train_forward(self, x):
        """
        Train the hybrid model components to predict Neutrophils.

        If not fine-tuning, then just train the NeutClassifier to produce neutrophils type based on image features

        If fine-tuning, then train the merging mechansim for the base Model features and hand crafted features to improve neutrophils prediction 

        Args:
            x (torch.Tensor): Input image batch

        Returns:
            x (torch.Tensor): Binary neutrophils prediction (0=SNE, 1=BNE)
        """

        # Don't do backprop for base model unless doing the final fine-tuning step
        with torch.no_grad():
            base_out = self.base_model(x)


        if self.fine_tuning:
            with torch.no_grad():
                morphological_features = self.neut_model(x)
        else:
            morphological_features = self.neut_model(x)

        if self.fine_tuning:
            # Project the base model output to the same dimensions as the morphological features
            base_proj = self.base_proj(self.base_model_features)

            # Gate the base model features and morphological features
            gate_input = torch.cat([base_proj, morphological_features], dim=1)

            gate_weight = self.gate(gate_input)

            print(f"gate  mean: {gate_weight.mean().item()}") # gives the average weight of features, typically ~0.85 so mostly uses Mamba features

            # Combine features with gate weight
            gated_features = gate_weight * base_proj + (1 - gate_weight) * morphological_features

            # Final classifier
            return self.classifier(gated_features).squeeze(-1)

        else: 
            # Final classifier
            return self.classifier(morphological_features).squeeze(-1)


    def test_forward(self, x):
        """
        Evaluate the hybrid model

        First get the base model predictions on the input batch
        For any samples classified as neutrophils by the base model, get a confirmed prediction using the trained neutrophils classifier
        Edit the base model outputs with the confirmed neutrophils classifications

        This aims to improve the neutrophils classifications with affeting classificatino performance on other classes

        Args:
            x (torch.Tensor): Input image batch

        Returns:
            x (torch.Tensor): WBC classification, with confirmed neutrophils classifications from the dedicated neutrophils classifier
        """

        # Get base model output
        base_out = self.base_model(x)
        
        # Get base model prediction
        base_preds = torch.argmax(base_out, dim=1)

        # Get images classified as neutrophils
        neut_preds = torch.isin(base_preds, self.neut_indices)
        
        # if no neutophils predictions return just return the base model output
        if not neut_preds.any():
            return base_out 

        # Get indices of images classified as neutrophils
        neut_pred_indices = torch.where(neut_preds)[0]

        # Get confirmed  type prediction
        morphological_features = self.neut_model(x[neut_pred_indices])

        # Project the base model output to the same dimensions as the morphological features
        base_proj = self.base_proj(self.base_model_features[neut_pred_indices])

        # Gate the base model features and morphological features
        gate_input = torch.cat([base_proj, morphological_features], dim=1)

        gate_weight = self.gate(gate_input)

        # print(f"gate  mean: {gate_weight.mean().item()}") # gives the average weight of features, typically ~0.85 so mostly uses Mamba features

        # Combine features with gate weight
        gated_features = gate_weight * base_proj + (1 - gate_weight) * morphological_features

        # Final classifier
        neut_type = self.classifier(gated_features)

        # Get binary classification of either BNE or SNE
        # E.g. if the base model predicts 3 samples are neutrophils the following will gives a confirmed binary SNE or BNE classification
        # tensor([[1],
        #         [1],
        #         [0]])
        neut_type = (torch.sigmoid(neut_type) > 0.5).float()  
        

        # Convert the binary neutrophils classification to the BNE/SNE index in n-class wbc classificaion
        # So the above becomes:
        # tensor([[2],  (BNE)
        #         [2],  (BNE)
        #         [0]]) (SNE)
        neut_preds = torch.where(neut_type == 1, self.BNE_index, self.SNE_index).squeeze(-1) # pred = 1 implies BNE, otherwise SNE

        # Create an n-class fake logits matrix so to insert the neutrophils predictions back into the original base model outputs matrix
        # So the above becomes:
        # tensor([[0., 0., 1., 0., 0., 0., 0., 0.],
        #         [0., 0., 1., 0., 0., 0., 0., 0.],
        #         [1., 0., 0., 0., 0., 0., 0., 0.]])
        neut_out = torch.zeros(base_out[neut_pred_indices].shape, device='cuda:0') 
        neut_out[torch.arange(neut_out.size(0), device='cuda:0'), neut_preds] = 1

        # Insert these classifications back into the batch's original logits
        # Producing output logits:
        # tensor([[0., 0., 1., 0., 0., 0., 0., 0.],     (Confirmed BNE)
        #         [0., 0., 1., 0., 0., 0., 0., 0.],     (Confirmed BNE)
        #         [0.1, 0.5, 0, 0.1, 0.4, 0., 0.5, 0],  (Some other class)
        #         [1., 0., 0., 0., 0., 0., 0., 0.]])    (Confirmed SNE)
        #         [0.1, 0.5, 0, 0.1, 0.4, 0., 0.5, 0]]) (Some other class)
        base_out[neut_pred_indices] = neut_out 

        return base_out


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

        return model

    else:
        # Initialise fresh model
        model = HybridClassifier(num_classes=num_classes, base_model=base_model)

    return model
