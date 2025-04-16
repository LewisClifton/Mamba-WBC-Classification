import torch

from models import init_model


def get_ensemble(ensemble_config, num_classes, device):

    base_models = []
    base_models_transforms = []

    for base_model_config in ensemble_config['models']:

        base_model, base_model_transform = init_model(base_model_config, num_classes, device)
        base_model.load_state_dict(torch.load(base_model_config['trained_model_path'], map_location=device), strict=False)

        base_model.eval()

        base_models.append(base_model)
        base_models_transforms.append(base_model_transform)

   

    if ensemble_config['ensemble_mode'] == 'stacking':

        meta_learner, _ = init_model(ensemble_config, num_classes, device)

        if 'meta_learner_path' in ensemble_config:
            meta_learner.load_state_dict(torch.load(ensemble_config['meta_learner_path'], map_location="cpu"))
            meta_learner = meta_learner.to(device)

        return meta_learner, base_models, base_models_transforms
    
    return None, base_models, base_models_transforms