
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms

from .vmamba import VSSM as vmamba


TRANSFORM_VMAMBA = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def __build_model(model_size='tiny'):

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


def get(num_classes, pretrained_model_path):

    # Get the correct URL
    model_size = 'tiny'
    if model_size == 'tiny':
        weights_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth"
    elif model_size == 'small':
        weights_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth"
    elif model_size == 'base':
        weights_url = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"

    # Build the model using the architecture specified
    model = __build_model(model_size)

    # Load pretrained weights if provided
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(weights_url, model_dir='models/vmamba/pretrained/', file_name=weights_url.split('/')[-1])['model']

    # Build the model from the pretrained
    model.classifier = nn.Sequential(
        OrderedDict([
            ("norm", model.classifier.norm),
            ("permute", model.classifier.permute),
            ("avgpool", model.classifier.avgpool),
            ("flatten", model.classifier.flatten),
            ("head", nn.Linear(model.num_features, state_dict["classifier.head.weight"].shape[0])),
    ]))
    model.load_state_dict(state_dict, strict=False)

    # Change model head
    model.classifier = nn.Sequential(
        OrderedDict([
            ("norm", model.classifier.norm),
            ("permute", model.classifier.permute),
            ("avgpool", model.classifier.avgpool),
            ("flatten", model.classifier.flatten),
            ("head", nn.Linear(model.num_features, num_classes)),
    ]))

    return model, TRANSFORM_VMAMBA