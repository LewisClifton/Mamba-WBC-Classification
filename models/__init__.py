import torch.nn as nn
from .transforms import TRANSFORMS

def init_model(config):
    """
    Initialise fresh model prior to training

    Args:
        config(dict): Training config containing model configuration

    Returns:
        torch.nn.Module: Initialised model ready for training
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """

    model_type = config['model']['type']
    num_classes=config['data']['n_classes']

    # Create the model and get the transform
    if model_type == 'swin':
        from models.swin import get_swin

        model = get_swin(num_classes)
        transform = TRANSFORMS['swin']

    elif model_type == 'medmamba':
        from .medmamba.medmamba import VSSM as MedMamba

        model = MedMamba(num_classes=num_classes)
        transform = TRANSFORMS['medmamba']

    elif model_type == 'vmamba':
        from .vmamba import get_vmamba

        model = get_vmamba(num_classes=num_classes)
        transform = TRANSFORMS['vmamba']

    elif model_type == 'mambavision':
        from .mambavision import get_mambavision
        
        model = get_mambavision(num_classes=num_classes)
        transform = TRANSFORMS['mambavision']

    elif model_type == 'vim':
        from .vim import get_vim

        model = get_vim()
    
    # Add new models using elif
    elif model_type == 'foo':
        pass

    

    return model, transform