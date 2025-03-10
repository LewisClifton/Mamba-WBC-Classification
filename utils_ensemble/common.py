from models import init_model
from models.ensemble import Ensemble

def get_ensemble(ensemble_config, num_classes, device):

    base_models = []
    base_models_transforms = []
    for base_model_config in ensemble_config['models']:
        base_model, base_model_transform = init_model(base_model_config, num_classes=None, device=device)

        base_models.append(base_model)
        base_models_transforms.append(base_model_transform)

    ensemble = Ensemble(base_models, num_classes, device)
    
    return ensemble, base_models, base_models_transforms