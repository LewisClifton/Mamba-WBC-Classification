import random
from PIL import ImageFilter, ImageOps

import torch
import torch.nn as nn
from timm.models.vision_transformer import _cfg
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from torchvision import transforms

from .models_mamba import VisionMamba


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


TRANSFORM_VIM = {
    'train': transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                gray_scale(p=1.0),
                Solarization(p=1.0),
                GaussianBlur(p=1.0)
            ]),
            transforms.ColorJitter(0.4, 0.4, 0.4),  # Assuming color_jitter = 0.4
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                 std=torch.tensor([0.229, 0.224, 0.225]))
        ]),
        
        'val': transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                 std=torch.tensor([0.229, 0.224, 0.225]))
        ])
}


def get(num_classes):

    # Build the model
    model_size = 'tiny'
    if model_size == 'tiny':
        weights_url = "https://huggingface.co/hustvl/Vim-tiny-midclstok/blob/main/vim_t_midclstok_76p1acc.pth"
        model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True)

    elif model_size == 'small':
        weights_url = "https://huggingface.co/hustvl/Vim-small-midclstok/blob/main/vim_s_midclstok_80p5acc.pth"
        model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True)

    elif model_size == 'base':
        weights_url = "https://huggingface.co/hustvl/Vim-base-midclstok/blob/main/vim_b_midclstok_81p9acc.pth"
        model = VisionMamba(
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True)

    model.default_cfg = _cfg()

    # Load the weights from the URL
    weights = torch.hub.load_state_dict_from_url(weights_url, model_dir='models/vim/pretrained/', file_name=weights_url.split('/')[-1])['model']
    model.load_state_dict(weights)

    # Edit final FC with correct number of classes
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model, TRANSFORM_VIM