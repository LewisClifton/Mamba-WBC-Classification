import torch

from models import init_model


def get_ensemble(ensemble_config, num_classes, device):
    """
    Initialise the ensemble using the specified configuration.

    Args:
        ensemble_config (dict): Ensemble configuration file
        num_classes (int): Number of dataset classes
        device (torch.device): Device to put models on
        num_classes (int): Number of classes in the dataset

    Returns:
        torch.Module: Loaded ensemble model.
    """

    base_models = {}
    base_models_transforms = {}

    # Load each base model and their transforms
    for base_model_config in ensemble_config['base_models']:

        base_model, base_model_transform = init_model(base_model_config, num_classes, device)
        base_model.load_state_dict(torch.load(base_model_config['trained_model_path'], map_location=device), strict=False)

        base_model.eval()

        base_models[base_model_config['name']] = base_model
        base_models_transforms[base_model_config['name']] = base_model_transform


    # Initialise the meta-learner if using stacking
    if ensemble_config['ensemble_mode'] == 'stacking':

        meta_learner, _ = init_model(ensemble_config, num_classes, device)
        
        # Load from pre-trained if required
        if 'meta_learner_path' in ensemble_config:
            meta_learner.load_state_dict(torch.load(ensemble_config['meta_learner_path'], map_location=device))

        meta_learner.eval()

        return meta_learner, base_models, base_models_transforms
    
    return None, base_models, base_models_transforms