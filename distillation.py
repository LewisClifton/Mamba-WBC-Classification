import argparse
import yaml
import time
from datetime import datetime
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_dataset, TransformedDataset, DistillationDataset
from utils.common import setup_dist, save, save_models
from models import init_model
from utils.distillation import train_loop
from models.student import Student


torch.backends.cudnn.enabled = True 


def train_model(student_model_config, dataset_config, train_dataset, device, out_dir, using_dist=True, verbose=False, save_val_outputs=False):
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

    # Initialise teacher model
    student_model = init_model(student_model_config, dataset_config['num_classes'], device)
    student_model = student_model.to(device)

    # Create data loaders and put model on device
    if using_dist: 
        train_sampler = DistributedSampler(train_dataset, shuffle=True)

        train_loader = DataLoader(train_dataset, batch_size=student_model_config['batch_size'], num_workers=1, sampler=train_sampler, drop_last=True)

        student_model = DDP(student_model, device_ids=[device], output_device=device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=student_model_config['batch_size'], shuffle=True, num_workers=1, drop_last=True)
        
    # Create criterion
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    criterion = nn.CrossEntropyLoss()

    # Create optimiser
    optimiser = optim.AdamW(student_model.parameters(), lr=student_model_config['learning_rate'], weight_decay=student_model_config['optim_weight_decay'])

    start_time = time.time()

    # Train loop
    trained, metrics = train_loop(student_model, student_model_config, train_loader, kl_criterion, criterion, optimiser, device, using_dist, verbose)

    if device in [0, 'cuda:0']:
        print('Done.')

    # Get runtime
    metrics['Time to train'] = time.time() - start_time

    if using_dist:
        return trained.module, metrics
    
    return trained, metrics
    

def main(rank, world_size, using_dist, out_dir, student_model_config, teacher_logits_path, dataset_config, dataset_download_dir, verbose=False):

    # Setup GPU network if required
    if using_dist: setup_dist(rank, world_size)

    # Train using specified dataset
    if dataset_config['name'] == 'chula':
        # Get dataset
        dataset = get_dataset(dataset_config, dataset_download_dir)
        dataset = DistillationDataset(dataset, dataset_config['num_classes'], teacher_logits_path)
        dataset = TransformedDataset(dataset)
        
        # Train
        trained, metrics = train_model(student_model_config, dataset_config, dataset, rank, out_dir, using_dist, verbose)
        save_models(out_dir, trained, student_model_config['name'], metrics)

    elif dataset_config['name'] == 'bloodmnist':
        # Get dataset
        train_dataset, val_dataset = get_dataset(dataset_config, dataset_download_dir)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

        # Train model only once (i.e. without k-fold cross validation)
        trained, metrics = train_model(student_model_config, dataset_config, train_dataset, val_dataset, rank, out_dir, using_dist, verbose)
        save_models(out_dir, trained, student_model_config['name'], metrics)

    # Can add more datasets here..
    elif dataset_config['name'] == 'foo':
        pass

    # Save model and log
    if dist.is_initialized():
        dist.barrier()
        
        if rank in [0, 'cuda:0']:
            save(out_dir, metrics, trained, student_model_config, dataset_config)

        dist.destroy_process_group()
    else:
        save(out_dir, metrics, trained, student_model_config, dataset_config)
        

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where trained model and log will be saved (default=cwd)', default='.')
    parser.add_argument('--student_model_config_path', type=str, help='Path to student model config .yml.', required=True)
    parser.add_argument('--teacher_logits_path', type=str, help='Path to teacher model config .yml.', required=True)
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset config .yml.', required=True)
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to be used for training. (default=2)', default=2)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Whether to print per epoch metrics during training')
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')
   
    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    student_model_config_path = args.student_model_config_path
    teacher_logits_path = args.teacher_logits_path
    dataset_config_path = args.dataset_config_path
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    verbose = args.verbose
    dataset_download_dir = args.dataset_download_dir

    # Get the model and dataset configs
    with open(student_model_config_path, 'r') as yml:
        student_model_config = yaml.safe_load(yml)
    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    # Set up output directory
    date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_{student_model_config['name']}')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create process group if using multi gpus on Linux
    using_dist = num_gpus != 1
    if using_dist:
        mp.spawn(main, args=(num_gpus, True, out_dir, student_model_config, teacher_logits_path, dataset_config, dataset_download_dir,  verbose), nprocs=num_gpus)
    else:
        main('cuda:0', 1, False, out_dir, student_model_config, teacher_logits_path, dataset_config, dataset_download_dir,  verbose)