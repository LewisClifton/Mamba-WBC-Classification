import torch

def init_model(model_config, num_classes):
    """
    Initialise fresh model prior to training

    Args:
        model_type (string): Model type e.g. 'swin'
        num_classes (int): Number of WBC classification classes

    Returns:
        torch.nn.Module: Initialised model ready for training
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """
    # TO DO: ADD DOCSTRINGS TO EACH MODEL INIT

    model_type = model_config['name']

    # Get the required model. Use inner imports to allow for different conda env / OS
    if model_type == 'swin':
        from .swin import get
        model, transform = get(num_classes=num_classes)

    elif model_type == 'medmamba':
        from .medmamba import get
        model, transform = get(num_classes=num_classes)

    elif model_type == 'vmamba':
        from .vmamba import get
        model, transform = get(num_classes=num_classes)

    elif model_type == 'mambavision':
        from .mambavision import get
        model, transform = get(num_classes=num_classes)

    elif model_type == 'vim':
        from .vim import get
        model, transform = get(num_classes=num_classes)

    # Add new models using elif
    elif model_type == 'foo':
        pass

    
    # Apply weights of pre-trained model if necessary
    if 'pretrained_model_path' in model_config.keys():
        model.load_state_dict(torch.load(model_config['pretrained_model_path']))

    return model, transform