import torch.nn as nn
from torchvision import transforms
import torch


from .classification.lib.models.local_vim import VisionMamba as LocalMamba
from .classification.lib.models.mamba.multi_mamba import MultiMamba
from timm.models.vision_transformer import _cfg


TRANSFORM_LOCALMAMBA = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=25),
        transforms.RandomHorizontalFlip(p=1.0), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


def __build_model(pretrained=False, **kwargs):
    directions = (
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'w2', 'w2_flip'],
        ['v', 'v_flip', 'w2', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h', 'h_flip', 'w2_flip', 'w7'],
        ['v', 'v_flip', 'w2', 'w2_flip'],
    )
    model = LocalMamba(
        patch_size=16, embed_dim=192, depth=20, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2", directions=directions, mamba_cls=MultiMamba, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


def get(num_classes, pretrained_model_path):

    model = __build_model()

    # Load pretrained weights first
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url("https://github.com/hunto/LocalMamba/releases/download/v1.0.0/local_vim_tiny.ckpt", model_dir="models/localmamba/pretrained/", file_name='vim_tiny')

    # Build the model from the pretrained
    pretrained_num_classes = state_dict["head.weight"].shape[0]
    
    model.head = nn.Linear(model.head.in_features, pretrained_num_classes)
    model.load_state_dict(state_dict, strict=False)

    # Change model head
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model, TRANSFORM_LOCALMAMBA