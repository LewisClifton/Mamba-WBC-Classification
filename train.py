from dataset import WBC5000dataset
from torch.utils .data import DataLoader
from torchvision import models, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import swin_t
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os

torch.backends.cudnn.enabled = True

images_path = "data/WBC 5000/"
labels_path = "data/labels.csv"


def train(model, n_epochs, train_loader, val_loader, criterion, optimizer, device):

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

        # Print epoch metrics]
        if device == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}]")
            print(f"Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {correct_train / total_train:.4f}")
            print(f"Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {correct_val / total_val:.4f}")

        return model, 

def main(rank, 
         world_size,
         images_path, 
         labels_path,
         n_epochs,
         lr,
         lr_weight_decay):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create model
    model = swin_t(weights="IMAGENET1K_V1")
    model.head = nn.Linear(model.head.in_features, 8)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=25),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=1.0), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset
    dataset = WBC5000dataset(images_path, labels_path)

    # Train/test split
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=lr_weight_decay)

    train(model, n_epochs, train_loader, val_loader, criterion, optimizer, rank)

    # Save model
    dist.barrier()
    if rank == 0:
        torch.save(model.module.state_dict(), "trained/")

    dist.destroy_process_group()

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, help='Path to directory for dataset containing training images', required=True)
    parser.add_argument('--labels_path', type=str, help='Path to labels.csv', required=True)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs to train with (default=20)', default=2)
    parser.add_argument('--lr', type=float, help='Learning rate (default=0.002)', default=0.002)
    parser.add_argument('--lr_weight_decay', type=float, help='Learning rate weight decay (default=2e-4)', default=2e-4)
    args = parser.parse_args()

    images_dir = args.images_dir
    labels_path = args.labels_path
    n_epochs = args.n_epochs
    lr = args.lr
    lr_weight_decay = args.lr_weight_decay

    device = 'cuda'
    num_gpus = 2
    
    mp.spawn(
            main,
            args=(num_gpus,
                  device,
                  images_dir,
                  labels_path,
                  n_epochs,
                  lr,
                  lr_weight_decay,),
            nprocs=num_gpus)