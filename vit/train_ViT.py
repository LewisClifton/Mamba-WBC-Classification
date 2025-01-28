import argparse

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.models import swin_t, swin_s, swin_b

from dataset import WBC5000dataset, TransformedDataset
from utils import *


torch.backends.cudnn.enabled = True

def train(model, n_epochs, train_loader, val_loader, criterion, optimizer, device):

    # Training metrics
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

            break

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
            print(f"Epoch [{epoch + 1}/{n_epochs}]")
            print(f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Combine the training metrics in a single dictionary
    metrics = {
            'Train accuracy per epoch' : train_accuracy_per_epoch,
            'Train loss per epoch' : train_loss_per_epoch,
            'Validation accuracy per epoch' : val_accuracy_per_epoch,
            'Validation loss per epoch' : val_loss_per_epoch,
        }
    
    # Average across GPUs if required
    if dist.is_initialized():
        def average_metric_across_gpus(metric):
            # Average the per epoch metrics across all gpu
            metric = torch.tensor(metric).to(device)
            dist.all_reduce(metric, op=dist.ReduceOp.AVG)
            return metric.tolist()
        
        metrics = {metric: average_metric_across_gpus(value) for metric, value in metrics.items()}

    return model, metrics

def main(rank, world_size, using_dist,
         images_dir, 
         labels_path,
         out_dir,
         hyperparameters,):

    # Setup GPU network if required
    if using_dist: setup_dist()

    # Create model
    if hyperparameters['ViT size'] == 'tiny':
        model = swin_t(weights="IMAGENET1K_V1")
    elif hyperparameters['ViT size'] == 'small':
        model = swin_s(weights="IMAGENET1K_V1")
    else:
        model = swin_b(weights="IMAGENET1K_V1")
    model.head = nn.Linear(model.head.in_features, 8)

    # Get dataset
    dataset = WBC5000dataset(images_dir, labels_path)

    # Train/test split
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=25),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=1.0), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = TransformedDataset(train_dataset, train_transform)
        
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = TransformedDataset(val_dataset, val_transform)

    # Set up data loaders and model (varies whether using single GPU Windows vs 2 GPU Linux)
    if dist.is_initialized():
        # Put model on this gpu
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # Distributed samplers split the dataset across the GPUs
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # Create data loaders for each of the samplers
        train_loader = DataLoader(train_dataset, batch_size=hyperparameters['Batch size'], shuffle=False, num_workers=8, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=hyperparameters['Batch size'], shuffle=False, num_workers=8, sampler=val_sampler)
        
    else:
        train_loader = DataLoader(train_dataset, batch_size=hyperparameters['Batch size'], shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=hyperparameters['Batch size'], shuffle=False, num_workers=0)

        model = model.to(rank)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['Learning rate'], weight_decay=hyperparameters['Optimizer weight decay'])

    # Train the model and get the training metrics
    trained, metrics = train(model, hyperparameters['Number of epochs'], train_loader, val_loader, criterion, optimizer, rank)

    # Save model and log
    if dist.is_initialized():
        dist.barrier()
        
        if rank == 0:
            save(trained.module.state_dict(), out_dir, metrics, hyperparameters)

        dist.destroy_process_group()
    else:
        save(trained.state_dict(), out_dir, metrics, hyperparameters)
        

        

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, help='Path to directory for dataset containing training images', required=True)
    parser.add_argument('--labels_path', type=str, help='Path to labels.csv', required=True)
    parser.add_argument('--out_dir', type=str, help='Path to directory where trained model and log will be saved', required=True)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs to train with (default=2)', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate (default=0.002)', default=0.002)
    parser.add_argument('--optim_weight_decay', type=float, help='Learning rate weight decay (default=2e-4)', default=2e-4)
    parser.add_argument('--using_windows', help='If using Windows machine for training (default=False)', action=argparse.BooleanOptionalAction)
    parser.add_argument('--vit_size', type=str, help='Use of "tiny", "small" or "base" SWIN transformer (default="tiny")', default='tiny', choices=['tiny', 'small', 'base'])
    parser.add_argument('--batch_size', type=int, help='Minibatch size (default=32)', default=32)
    args = parser.parse_args()

    images_dir = args.images_dir
    labels_path = args.labels_path
    out_dir = args.out_dir
    using_windows = args.using_windows

    hyperparameters = {
        'Learning rate' : args.lr,
        'Optimizer weight decay' : args.optim_weight_decay,
        'Batch size' : args.batch_size,
        'ViT size' : args.vit_size,
        'Number of epochs' : args.n_epochs,
    }
    
    # Either using windows with one GPU (test of my laptop) or Linus with two GPUs (BC4)
    if using_windows:
        main('cuda', 1, False, # GPU setup arguments
             images_dir,
             labels_path,
             out_dir,
             hyperparameters,)
    else:
        mp.spawn(main,
                 args=(2, True, # GPU setup arguments
                       images_dir,
                       labels_path,
                       out_dir,
                       hyperparameters,),
                 nprocs=2)