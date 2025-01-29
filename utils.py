import os
from datetime import datetime
import json

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.models import swin_t, swin_s, swin_b
from torchvision import transforms


def write_dict_to_file(file, dict_):
    # For writing dictionary contents to a file when saving
    for k, v in dict_.items():
        if isinstance(v, list) and len(v) == 0: continue # avoids error
        file.write(f'{k}: {v}\n')


def save_models(out_dir, all_trained, using_dist):
    # Create trained models directory if needed
    model_dir = os.path.join(out_dir, 'trained/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save trained models
    for idx, model in enumerate(all_trained):
        model_path = os.path.join(model_dir, f'ViT_fold_{idx}.pth')
        if using_dist: 
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
    print(f'\n{len(all_trained)} trained models saved to {model_dir}')


def save_log(out_dir, all_metrics, config, date):
    # Save training log
    log_path = os.path.join(out_dir, 'log.txt')
    with open(log_path , 'w') as file:
        
        file.write('Model configuration:\n')
        file.write(f'Date/time of creation: {date}\n')
        write_dict_to_file(file, config)

        for idx, fold_metric in enumerate(all_metrics):
            file.write(f'\nFold {idx} training metrics:\n')
            write_dict_to_file(file, fold_metric)

        
    print(f'\nSaved log to {log_path}')

def save_config(out_dir, config):
    # Save model configuration
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

def save(out_dir, all_metrics, all_trained, config, using_dist):
    # Get date/time of saving
    date = datetime.now().strftime('%Y_%m_%d_%p%I_%M')

    # Create output directory
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Save models, log and config json
    save_models(out_dir, all_trained, using_dist)
    save_log(out_dir, all_metrics, config, date)
    save_config(out_dir, config)


def setup_dist(rank, world_size):
     # Set up process group and gpu model
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


# Data transforms
TRANSFORMS = {
    'swin': {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=25),
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=1.0), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        },

    'medmamba': {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    
}

def init_model(config):

    model_type = config['Model type']
    # Create model
    if model_type == 'swin_tiny':
        model = swin_t(weights='IMAGENET1K_V1')
    elif model_type == 'swin_small':
        model = swin_s(weights='IMAGENET1K_V1')
    elif model_type == 'swin_base':
        model = swin_b(weights='IMAGENET1K_V1')
    elif model_type == 'medmamba':
        from medmamba.medmamba import VSSM as MedMamba # Import here as inner imports don't work on windows
        model = MedMamba(num_classes=config['Number of classes'])

    if 'swin' in model_type:
        model.head = nn.Linear(model.head.in_features, 8)
        transform = TRANSFORMS['swin']
    elif model_type == 'medmamba':
        transform = TRANSFORMS['medmamba']

    return model, transform

def average_across_gpus(list_, device):
    metric_tensor = torch.tensor(list_).to(device)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
    return metric_tensor.tolist()
