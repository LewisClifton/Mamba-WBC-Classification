def select_model(model_config, num_classes, device):
    """
    Initialise a model which can class num_classes classes based on the provided configuration file. 

    Args:
        model_config (dict): Model configuration file
        num_classes (int): Number of classes in the dataset
        device (torch.device): Device to place model on

    Returns:
        nn.Module: Initialised model
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """

    # Get the required model. Use inner imports to limit importing and prevent breaking when packages are missing

    model_type = model_config['name']
    pretrained_model_path = model_config['pretrained_model_path'] if 'pretrained_model_path' in model_config.keys() else None

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

    elif model_type == 'localmamba':
        from .localmamba import get
        model, transform = get(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'hybrid':
        from .hybrid import get

        base_model_config = model_config['base_model_config']
        base_model, transform = select_model(model_config=base_model_config, num_classes=8, device=device)

        model = get(pretrained_model_path=pretrained_model_path, num_classes=num_classes, base_model=base_model)
        
        transform = None

    elif model_type == 'meta_learner':
        from .meta_learner import get

        model, transform = get(num_base_models=model_config['n_base_models'], num_classes=num_classes, pretrained_model_path=pretrained_model_path)


    model = model.to(device)

    return model, transform


def init_model(model_config, num_classes, device):
    """
    Initialise fresh model prior to training

    Args:
        model_config (dict): Model configuration
        num_classes (int): Number of WBC classes to predict

    Returns:
        torch.nn.Module: Initialised model ready for training
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """
    # TO DO: ADD DOCSTRINGS TO EACH MODEL INIT
    
    if 'use_improvements' in model_config.keys():
        pass
    else: 
        model, transform = select_model(model_config, num_classes, device)
    
    model = model.to(device)

    return model, transform