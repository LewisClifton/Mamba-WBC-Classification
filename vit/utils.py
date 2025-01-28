import os
from datetime import datetime
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.models import swin_t, swin_s, swin_b


def write_dict_to_file(file, dict_):
    # For writing dictionary contents to a file when saving
    for k, v in dict_.items():
        if isinstance(v, list) and len(v) == 0: continue # avoids error
        file.write(f'{k}: {v}\n')


def save_models(out_dir, all_trained, using_dist):
    # Save trained models
    model_dir = os.path.join(out_dir, 'trained/')
    for model, idx in enumerate(all_trained):
        model_path = os.path.join(model_dir, f'ViT_fold_{idx}.pth')
        if using_dist: 
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
    print(f'{len(all_trained)} trained models saved to {model_dir}.')


def save_log(out_dir, all_metrics, model_config, date):
    # Save training log
    log_path = os.path.join(out_dir, 'log.txt')
    with open(log_path , 'w') as file:
        
        file.write(f'Date/time of creation: {date}\n')

        for fold_metric, idx in enumerate(all_metrics):
            file.write(f'\nFold {idx} training metrics:\n')
            write_dict_to_file(file, fold_metric)

        file.write('\nModel configuration:\n')
        write_dict_to_file(file, model_config)
    print(f'Saved log to {log_path}.')


def save(out_dir, all_metrics, all_trained, model_config, using_dist):
    # Get date/time of saving
    date = datetime.now().strftime("%Y_%m_%d_%p%I_%M")

    # Create output directory
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Save models and log
    save_models(out_dir, all_trained, using_dist)
    save_log(out_dir, all_metrics, model_config, date)


def setup_dist(rank, world_size):
     # Set up process group and gpu model
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def init_model(model_config):
    # Create model
    if model_config['ViT size'] == 'tiny':
        model = swin_t(weights="IMAGENET1K_V1")
    elif model_config['ViT size'] == 'small':
        model = swin_s(weights="IMAGENET1K_V1")
    else:
        model = swin_b(weights="IMAGENET1K_V1")
    model.head = nn.Linear(model.head.in_features, 8)

    return model

def average_across_gpus(list_, device):
    metric_tensor = torch.tensor(list_).to(device)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
    return metric_tensor.tolist()
