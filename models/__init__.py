from .transforms import get_transform

def init_model(model_type, num_classes):
    """
    Initialise fresh model prior to training

    Args:
        config(dict): Training config containing model configuration

    Returns:
        torch.nn.Module: Initialised model ready for training
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """


    # Get the required model
    if model_type == 'swin':
        from models.swin import get_swin

        model = get_swin(num_classes=num_classes)

    elif model_type == 'medmamba':
        from .medmamba import get_medmamba

        model = get_medmamba(num_classes=num_classes)

    elif model_type == 'vmamba':
        from .vmamba import get_vmamba

        model = get_vmamba(num_classes=num_classes)

    elif model_type == 'mambavision':
        from .mambavision import get_mambavision
        
        model = get_mambavision(num_classes=num_classes)

    elif model_type == 'vim':
        from .vim import get_vim

        model = get_vim(num_classes=num_classes)

    # Add new models using elif
    elif model_type == 'foo':
        pass

    transform = get_transform(model_type)

    return model, transform