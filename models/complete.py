from . import init_model

import torch
import torch.nn as nn

# Two-tier model making use of a bespoke neutrophils classifier to try to improve results

class CompleteClassifier(nn.Module):
    def __init__(self, model_config, dataset_config):
        super().__init__()

        # Load the wbc classifier
        self.wbc_model, self.model_transforms = init_model(model_config, dataset_config['n_classes'])
        self.wbc_model.load_state_dict(torch.load(model_config['trained_model_path'], map_location='cpu'))
        self.wbc_model.eval()

        # Get the indices of the SNE and BNE neutrophils classes
        self.BNE_index = dataset_config['classes'].index('BNE')
        self.SNE_index = dataset_config['classes'].index('SNE')
        self.neutrophils_indices = torch.tensor([self.BNE_index, self.SNE_index])

        # Load the neutrophils classifier
        model_config['trained_model_path'] = model_config['neutrophil_model_path']
        self.neutrophils_model, _ = init_model(model_config, num_classes=2)
        self.neutrophils_model.load_state_dict(torch.load(model_config['neutrophil_model_path'], map_location='cpu'))
        self.neutrophils_model.eval()

    def forward(self, x):
        wbc_out = self.wbc_model(x)
        wbc_type = torch.argmax(wbc_out, dim=1)

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
        
        return wbc_type