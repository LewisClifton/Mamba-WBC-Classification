import torch

from models import init_model
from models.meta_learner import MetaLearner


def get_meta_learner(meta_learner_config, num_classes, device):

    base_models = []
    base_models_transforms = []

    stacking = False#meta_learner_config['meta_learner_mode'] == 'stacking'

    for base_model_config in meta_learner_config['models']:

        if stacking: 
            base_model_config['pretrained_model_path'] = base_model_config['trained_model_path']
            base_model, base_model_transform = init_model(base_model_config, None, device)

        else:

            base_model, base_model_transform = init_model(base_model_config, num_classes, device)
            base_model.load_state_dict(torch.load(base_model_config['trained_model_path'], map_location=device), strict=False)

        base_model.eval()

        base_models.append(base_model)
        base_models_transforms.append(base_model_transform)

    meta_learner = meta_learner(base_models, num_classes, device)

    if 'stacking_model_path' in meta_learner_config:
        meta_learner.load_state_dict(torch.load(meta_learner_config['stacking_model_path'], map_location="cpu"))

    meta_learner = meta_learner.to(device)
    
    return meta_learner, base_models, base_models_transforms