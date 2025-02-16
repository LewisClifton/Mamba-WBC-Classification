
import torch
import yaml

from vmamba import VSSM as vmamba

def build_model(num_classes, vssm_config):
    return vmamba(
        num_classes = num_classes,
        drop_path_rate = vssm_config.MODEL.DROP_PATH_RATE,
        dims = vssm_config.MODEL.VSSM.EMBED_DIM,
        depths = vssm_config.MODEL.VSSM.DEPTHS,
        ssm_d_state = vssm_config.MODEL.VSSM.SSM_D_STATE,
        ssm_dt_rank = vssm_config.MODEL.VSSM.SSM_DT_RANK,
        ssm_ratio = vssm_config.MODEL.VSSM.SSM_RATIO,
        ssm_conv = vssm_config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias = vssm_config.MODEL.VSSM.SSM_CONV_BIAS,
        forward_type = vssm_config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio = vssm_config.MODEL.VSSM.MLP_RATIO,
        downsample_version = vssm_config.MODEL.VSSM.DOWNSAMPLE,
        pachembed_version = vssm_config.MODEL.VSSM.PATCHEMBED,
        norm_layer = vssm_config.MODEL.VSSM.NORM_LAYER,
    )

def get_vmamba(num_classes):

    config_path = 'pretrained/vmambav2v_tiny_224.yaml'
    weights_path = 'pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth'

    with open(config_path, 'r') as yml:
        vssm_config = yaml.safe_load(yml)  

    model = build_model(num_classes, vssm_config)
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    return model