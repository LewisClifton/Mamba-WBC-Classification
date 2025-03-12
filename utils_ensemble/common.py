import torch

from models import init_model
from models.ensemble import Ensemble


def get_ensemble(ensemble_config, num_classes, device):

    base_models = []
    base_models_transforms = []

    stacking = 'stacking_model_path' in ensemble_config

    for base_model_config in ensemble_config['models']:

        if stacking: 
            base_model_config['pretrained_model_path'] = base_model_config['trained_model_path']
            base_model, base_model_transform = init_model(base_model_config, None, device)

        else:

            base_model, base_model_transform = init_model(base_model_config, num_classes, device)
            base_model.load_state_dict(torch.load(base_model_config['trained_model_path'], map_location=device), strict=False)

        base_model.eval()

        base_models.append(base_model)
        base_models_transforms.append(base_model_transform)

    ensemble = Ensemble(base_models, num_classes, device)

    if 'stacking_model_path' in ensemble_config:
        ensemble.load_state_dict(torch.load(ensemble_config['stacking_model_path'], map_location="cpu"))

    ensemble = ensemble.to(device)
    
    return ensemble, base_models, base_models_transforms