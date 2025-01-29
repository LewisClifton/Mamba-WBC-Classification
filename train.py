import argparse
import json

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from data.dataset import WBC5000dataset, TransformedDataset
from utils import *


torch.backends.cudnn.enabled = True


DEFAULT_config = {
    'Model type': 'swin_t',
    'Number of classes': 6,
    'WBC classes': ["BNE", "SNE", "Basophil", "Eosinophil", "Monocyte", "Lymphocyte"],
    'Learning rate' : 0.002,
    'Optimizer weight decay' : 2e-4,
    'Batch size' : 32,
    'Number of epochs' : 1,
    'Number of K-fold cross validation folds' : 5,
}


def train_fold(model, train_loader, val_loader, n_epochs, criterion, optimizer, device, using_dist):
    """
    Train and evaluate the model for a single fold.
    """
    train_accuracy_per_epoch = []
    train_loss_per_epoch = []
    val_accuracy_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        # Store metrics for this epoch
        train_accuracy_per_epoch.append(train_accuracy)
        train_loss_per_epoch.append(avg_train_loss)
        val_accuracy_per_epoch.append(val_accuracy)
        val_loss_per_epoch.append(avg_val_loss)

        # Print epoch metrics
        if device == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}]')
            print(f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')


    # Return final metrics
    if using_dist:
        # Average per epoch metrics across over all GPUs
        return model, {
            'Average Train Accuracy per epoch': average_across_gpus(train_accuracy_per_epoch, device),
            'Average Train Loss per epoch': average_across_gpus(train_loss_per_epoch, device),
            'Average Validation Accuracy per epoch': average_across_gpus(val_accuracy_per_epoch, device),
            'Average Validation Loss per epoch': average_across_gpus(val_loss_per_epoch, device)
        }
    
    return model, {
        'Train Accuracy per epoch': train_accuracy_per_epoch,
        'Train Loss per epoch': train_loss_per_epoch,
        'Validation Accuracy per epoch': val_accuracy_per_epoch,
        'Validation Loss per epoch': val_loss_per_epoch
    }


def train_kfolds(config, dataset, device, using_dist=True):
    """
    Perform K-fold cross-validation (using Distributed Data Parallel if required).
    """
    print(f'\nTraining {config["Model type"]} for {config["Number of K-fold cross validation folds"]} folds.')

    # List of 5 dicts (1 for each fold), containing metrics for each fold
    all_metrics = []

    # Trained models for each fold
    all_trained = []

    # 5-Fold cross-validation setup
    folds = KFold(n_splits=config['Number of K-fold cross validation folds'], shuffle=True, random_state=42)

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(folds.split(dataset)):
        print(f'\nFold {fold + 1}/{5}...')

        # Reinitialise model
        model, model_transforms = init_model(config)

        # Create train and validation subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Apply transforms
        train_dataset = TransformedDataset(train_subset, model_transforms['train'])
        val_dataset = TransformedDataset(val_subset, model_transforms['val'])

        # Create data loaders and put model on device
        if using_dist:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)

            train_loader = DataLoader(train_dataset, batch_size=config['Batch size'], num_workers=8, sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=config['Batch size'], num_workers=8, sampler=val_sampler)

            model = DDP(model, device_ids=[device], output_device=device)
        else:
            train_loader = DataLoader(train_dataset, batch_size=config['Batch size'], shuffle=False, num_workers=8)
            val_loader = DataLoader(val_dataset, batch_size=config['Batch size'], shuffle=False, num_workers=8)

            model = model.to(device)
        
        # Create criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['Learning rate'], weight_decay=config['Optimizer weight decay'])

        # Train the model
        trained, metrics = train_fold(model, train_loader, val_loader, config['Number of epochs'], criterion, optimizer, device, using_dist)
        
        all_metrics.append(metrics)
        all_trained.append(trained)

        print('\nDone.\n')


    return all_metrics, all_trained


def main(rank, using_dist,
         images_dir, 
         labels_path,
         out_dir,
         config,):

    # Setup GPU network if required
    if using_dist: setup_dist()

    # Get dataset
    dataset = WBC5000dataset(images_dir, labels_path, wbc_types=config['WBC classes'])

    # Train the model and get the training metrics for each fold
    all_metrics, all_trained = train_kfolds(config, dataset, rank, using_dist)

    # Save model and log
    if dist.is_initialized():
        dist.barrier()
        
        if rank == 0:
            save(out_dir, all_metrics, all_trained, config, using_dist=True)

        dist.destroy_process_group()
    else:
        save(out_dir, all_metrics, all_trained, config, using_dist=False)
        

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, help='Path to directory for dataset containing training images', required=True)
    parser.add_argument('--labels_path', type=str, help='Path to labels.csv', required=True)
    parser.add_argument('--out_dir', type=str, help='Path to directory where trained model and log will be saved (default=cwd)', required=True)
    parser.add_argument('--config_path', type=str, help='Path to model config .json. If config file unprovided, default values are used. Missing fields are replaced with default values')
    parser.add_argument('--using_windows', action=argparse.BooleanOptionalAction, help='If using Windows machine for training. Forces --num_gpus to 1 (default=False)')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to be used for training. (default=2)', default=2)
    parser.add_argument('--num_val_folds', type=int, help='Number of folds for K-fold cross validation (default=5)', default=5)
    
    # Parse command line args
    args = parser.parse_args()
    images_dir = args.images_dir
    labels_path = args.labels_path
    out_dir = args.out_dir
    config_path = args.config_path
    using_windows = args.using_windows
    num_gpus = args.num_gpus
    num_val_folds = args.num_val_folds

    # Multi GPU not supported for windows and trivially not for 1 GPU
    using_dist = True
    if using_windows or num_gpus == 1:
        using_dist = False

    # Get the model config and fill in missing keys with default values (defined at top of this file)
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    data_with_defaults = {key: config.get(key, DEFAULT_config.get(key)) for key in DEFAULT_config}
    
    # Create process group if using multi gpus on Linux
    if using_dist:
        mp.spawn(main,
                 args=(True,
                       images_dir,
                       labels_path,
                       out_dir,
                       config,),
                 nprocs=2)
    else:
        main('cuda', False,
             images_dir,
             labels_path,
             out_dir,
             config,)