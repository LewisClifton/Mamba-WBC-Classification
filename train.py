import argparse
import yaml

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from data.dataset import WBC5000dataset, BloodMNIST, TransformedDataset
from utils import *
from models import init_model


torch.backends.cudnn.enabled = True


def train_kfolds(config, dataset, device, using_dist=True, verbose=False):
    """
    Train a model with k-fold cross-validation

    Args:
        config(dict): Dictionary containing training configuration with top level keys of "model","data","params"
        dataset(torch.utils.data.Dataset): Dataset used for training
        device(torch.cuda.device): Device used for training
        using_dist(bool): True if using distributed training across 2+ GPUs

    Returns:
        list[torch.Module]: List of trained models, one for each fold
        list[dict]: List of training metrics for each of the models
    """
    if device in [0, 'cuda:0']:
        print(f'\nTraining {config["model"]["type"]} for {config["model"]["k-folds"]} folds.')

    # List of 5 dicts (1 for each fold), containing metrics for each fold
    all_metrics = []

    # Trained models for each fold
    all_trained = []

    # 5-Fold cross-validation setup
    folds = KFold(n_splits=config['model']['k-folds'], shuffle=True, random_state=42)

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(folds.split(dataset)):
        if device in [0, 'cuda:0']:
            print(f'\nFold {fold + 1}/{5}:')
        
        # Create train and validation subsets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        # Train this fold
        trained, metrics = train_model(config, train_dataset, val_dataset, device, using_dist, verbose)

        # Aggregate models and metrics from this fold
        all_trained.append(trained)
        all_metrics.append(metrics)

    return all_trained, all_metrics

def train_model(config, train_dataset, val_dataset, device, using_dist=True, verbose=False):
    """
    Train a single model

    Args:
        config(dict): Dictionary containing training configuration with top level keys of "model","data","params"
        train_dataset(torch.utils.data.Dataset): Dataset used for training
        val_dataset(torch.utils.data.Dataset): Dataset used for validation
        device(torch.cuda.device): Device used for training
        using_dist(bool): True if using distributed training across 2+ GPUs

    Returns:
        torch.Module: Trained model
        dict: Training metrics for the model
    """
    if device in [0, 'cuda:0']:
        print('Training...')
    # Initialise model
    model, model_transforms = init_model(config['model']['type'], config['data']['n_classes'])
    model = model.to(device)

    # Apply transforms
    train_dataset = TransformedDataset(train_dataset, model_transforms['train'])
    val_dataset = TransformedDataset(val_dataset, model_transforms['val'])

    # Create data loaders and put model on device
    if using_dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=config['params']['batch_size'], num_workers=1, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=config['params']['batch_size'], num_workers=1, sampler=val_sampler)

        model = DDP(model, device_ids=[device], output_device=device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['params']['batch_size'], shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=config['params']['batch_size'], shuffle=False, num_workers=1)
    
    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['params']['learning_rate'], weight_decay=config['params']['optim_weight_decay'])

    # Train the model
    trained, metrics = train_loop(model, train_loader, val_loader, config['params']['epochs'], criterion, optimizer, device, using_dist, verbose)

    if device in [0, 'cuda:0']:
        print('Done.')

    if using_dist:
        return trained.module, metrics
    
    return trained, metrics
    

def main(rank, world_size, using_dist, out_dir, config, verbose=False):

    # Setup GPU network if required
    if using_dist: setup_dist(rank, world_size)

    # Train using specified dataset with/without k-fold cross validation
    if config['data']['dataset'] == 'chula':
        # Get dataset
        dataset = WBC5000dataset(config['data']['images_dir'], config['data']['labels_path'], wbc_types=config['data']['classes'])

        # Train the model using k-fold cross validation and get the training metrics for each fold
        trained, metrics = train_kfolds(config, dataset, rank, using_dist, verbose)

    elif config['data']['dataset'] == 'bloodmnist':
        # Get dataset
        train_dataset = BloodMNIST(split='train', download=True, size=224)
        val_dataset = BloodMNIST(split='val', download=True, size=224)

        # Train model only once (i.e. without k-fold cross validation)
        trained, metrics = train_model(config, train_dataset, val_dataset, rank, using_dist, verbose)

    # Can add more datasets here..
    elif config['data']['dataset'] == 'foo':
        pass


    # Save model and log
    if dist.is_initialized():
        dist.barrier()
        
        if rank in [0, 'cuda:0']:
            save(out_dir, metrics, trained, config, using_dist=True)

        dist.destroy_process_group()
    else:
        save(out_dir, metrics, trained, config, using_dist=False)
        

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where trained model and log will be saved (default=cwd)', default='.')
    parser.add_argument('--config_path', type=str, help='Path to model config .yml.', required=True)
    parser.add_argument('--using_windows', action=argparse.BooleanOptionalAction, help='If using Windows machine for training. Forces --num_gpus to 1')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to be used for training. (default=2)', default=2)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Whether to print per epoch metrics during training')
   
    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    config_path = args.config_path
    using_windows = args.using_windows
    num_gpus = args.num_gpus
    verbose = args.verbose

    # Get the model config and fill in missing keys with default values (defined at top of this file)
    with open(config_path, 'r') as yml:
        config = yaml.safe_load(yml)

    # Multi GPU not supported for windows and trivially not for 1 GPU
    using_dist = True
    if using_windows or num_gpus == 1:
        using_dist = False

    # Create process group if using multi gpus on Linux
    if using_dist:
        mp.spawn(main, args=(num_gpus, True, out_dir, config, verbose), nprocs=num_gpus)
    else:
        main(0, 1, False, out_dir, config, verbose)