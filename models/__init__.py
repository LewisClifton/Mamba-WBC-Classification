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

    # Create the model and get the transform
    if model_type == 'swin_t':
        from torchvision.models import swin_t

        model = swin_t(weights='IMAGENET1K_V1')
        model.head = nn.Linear(model.head.in_features, config['data']['n_classes'])
        transform = TRANSFORMS['swin']

    elif model_type == 'swin_s':
        from torchvision.models import swin_s

        model = swin_s(weights='IMAGENET1K_V1')
        model.head = nn.Linear(model.head.in_features, config['data']['n_classes'])
        transform = TRANSFORMS['swin']

    elif model_type == 'swin_b':
        from torchvision.models import swin_b

        model = swin_b(weights='IMAGENET1K_V1')
        model.head = nn.Linear(model.head.in_features, config['data']['n_classes'])
        transform = TRANSFORMS['swin']

    elif model_type == 'medmamba':
        from models.medmamba import VSSM as MedMamba

        model = MedMamba(num_classes=config['data']['n_classes'])
        transform = TRANSFORMS['medmamba']

    elif model_type == 'vmamba':
        from vmamba import get_vmamba

        model = get_vmamba(num_classes=config['data']['n_classes'])
        transform = TRANSFORMS['vmamba']

    elif model_type == 'mambavision':
        from transformers import AutoModelForImageClassification
        
        # Need a wrapper to remove the dict wrapping of mambavision.forward
        class MambaVisionWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)

            def forward(self, x):
                return self.model(x)['logits']


        model = MambaVisionWrapper()
        transform = TRANSFORMS['mambavision']

    elif model_type == 'vim':
        pass
    
    # Add new models using elif
    elif model_type == 'foo':
        pass

    

    return model, transform