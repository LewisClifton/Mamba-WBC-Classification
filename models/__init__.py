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
    pretrained_model_path = model_config['pretrained_model_path'] if 'pretrained_model_path' in model_config.keys() else None

    # Get the required model. Use inner imports to allow for different conda env / OS
    if model_type == 'swin':
        from .swin import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'medmamba':
        from .medmamba import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'vmamba':
        from .vmamba import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'mambavision':
        from .mambavision import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'vim':
        from .vim import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'plainmamba':
        from .plainmamba import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'localmamba':
        from .localmamba import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    # Add new models using elif
    elif model_type == 'foo':
        pass

    return model, transform