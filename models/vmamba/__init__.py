
import torch
import torch.nn as nn
import yaml

from .vmamba import VSSM as vmamba

SMALL_URL = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth"
TINY_URL = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth"
BASE_URL = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"

def build_model(num_classes, model_size='tiny'):

    model_size = 'tiny'
    if model_size == 'tiny':
        return vmamba(
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v05_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln2d", 
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224, 
        )
    elif model_size == 'small':
        return vmamba(
            depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v05_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln2d", 
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224, 
        )
    elif model_size == 'base':
        return vmamba(
            depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v05_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln2d", 
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224, 
        )


def get_vmamba(num_classes):

    # Get the correct URL
    model_size = 'tiny'
    if model_size == 'tiny':
        weights_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth"
    elif model_size == 'small':
        weights_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth"
    elif model_size == 'base':
        weights_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"

    # Build the model using the architecture specified
    model = build_model(num_classes, model_size)

    # Load the weights from the URL
    weights = torch.hub.load_state_dict_from_url(weights_url, model_dir='models/vmamba/pretrained/', file_name=weights_url.split('/')[-1])['model']
    model.load_state_dict(weights)

    # Edit final FC with correct number of classes
    model.classifier.head=nn.Linear(model.num_features, num_classes),

    return model