import torch

from models import init_model
from models.ensemble import Ensemble

def get_ensemble(ensemble_config, num_classes, device):

    base_models = []
    base_models_transforms = []
    for base_model_config in ensemble_config['models']:
        base_model, base_model_transform = init_model(base_model_config, num_classes=None, device=device)
        base_model.eval()

        base_models.append(base_model)
        base_models_transforms.append(base_model_transform)

    ensemble = Ensemble(base_models, num_classes, device)

    if 'stacking_model_path' in ensemble_config:
        ensemble.load_state_dict(torch.load(ensemble_config['stacking_model_path'], map_location="cpu"))

    ensemble = ensemble.to(device)
    
    return ensemble, base_models, base_models_transforms