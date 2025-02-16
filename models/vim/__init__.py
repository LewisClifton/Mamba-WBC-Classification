import torch
import torch.nn as nn
from timm.models.vision_transformer import _cfg

from .vim.models_mamba import VisionMamba


def get_vim(num_classes):

    # Build the model
    model_size = 'tiny'
    if model_size == 'tiny':
        weights_url = "https://huggingface.co/hustvl/Vim-tiny-midclstok/blob/main/vim_t_midclstok_76p1acc.pth"
        model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True)

    elif model_size == 'small':
        weights_url = "https://huggingface.co/hustvl/Vim-small-midclstok/blob/main/vim_s_midclstok_80p5acc.pth"
    
    elif model_size == 'base':
        weights_url = "https://huggingface.co/hustvl/Vim-base-midclstok/blob/main/vim_b_midclstok_81p9acc.pth"

    # Build the model using the architecture specified
    model.default_cfg = _cfg()

    # Load the weights from the URL
    weights = torch.hub.load_state_dict_from_url(weights_url, model_dir='models/vim/pretrained/', file_name=weights_url.split('/')[-1])['model']
    model.load_state_dict(weights)

    # Edit final FC with correct number of classes
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model