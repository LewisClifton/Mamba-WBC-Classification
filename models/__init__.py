def select_model(model_config, num_classes, device):
    # Get the required model. Use inner imports to allow for different conda env / OS

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

    elif model_type == 'complete':
        from .neutrophils import get
    
        base_model_config = model_config['base_model_config']
        base_model_config['num_classes'] = None # so head is removed
        base_model, transform = select_model(base_model_config, None, device)

        model = get(base_model=base_model, num_classes=num_classes, pretrained_model_path=pretrained_model_path)

    elif model_type == 'meta_learner':
        from .meta_learner import get

        model, transform = get(num_base_models=model_config['num_base_models'], num_classes=num_classes, pretrained_model_path=pretrained_model_path)


    model = model.to(device)

    return model, transform


def init_model(model_config, num_classes, device):
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
    
    if 'use_improvements' in model_config.keys():
        pass
    else: 
        model, transform = select_model(model_config, num_classes, device)
    
    model = model.to(device)

    return model, transform