import os
from datetime import datetime
import yaml

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


def save_models(out_dir, trained, using_dist):
    # Create trained models directory if needed
    model_dir = os.path.join(out_dir, 'trained/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if isinstance(trained, list):
        # Save trained models for each fold
        for idx, model in enumerate(trained):
            model_path = os.path.join(model_dir, f'SWIN_ViT_fold_{idx}.pth')
            if using_dist: 
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        print(f'\n{len(trained)} trained models saved to {model_dir}')
    else:
        # Save the model
        model_path = os.path.join(model_dir, f'SWIN_ViT.pth')
        if using_dist: 
            torch.save(trained.module.state_dict(), model_path)
        else:
            torch.save(trained.state_dict(), model_path)


def save_log(out_dir, metrics, date):
    # Save training log
    log_path = os.path.join(out_dir, 'log.txt')
    with open(log_path , 'w') as file:

        file.write(f'Date/time of creation: {date}\n')
        if isinstance(metrics, list):
            # Save metrics for each fold
            for idx, fold_metrics in enumerate(metrics):
                file.write(f'\nFold {idx} training metrics:\n')
                write_dict_to_file(file, fold_metrics)
        else:
            # Save metrics for the model
            write_dict_to_file(file, metrics)

        
    print(f'\nSaved log to {log_path}')

def save_config(out_dir, config):
    # Save model configuration
    config_path = os.path.join(out_dir, 'config.yml')
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

def save(out_dir, metrics, trained, config, using_dist):
    # Get date/time of saving
    date = datetime.now().strftime('%Y_%m_%d_%p%I_%M')

    # Create output directory
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Save models, log and config yml
    save_models(out_dir, trained, using_dist)
    save_log(out_dir, metrics, date)
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

    model_type = config['model']['type']
    # Create model
    if model_type == 'swin_t':
        model = swin_t(weights='IMAGENET1K_V1')
    elif model_type == 'swin_s':
        model = swin_s(weights='IMAGENET1K_V1')
    elif model_type == 'swin_b':
        model = swin_b(weights='IMAGENET1K_V1')
    elif model_type == 'medmamba':
        from models.medmamba import VSSM as MedMamba # Import here as inner imports don't work on windows
        model = MedMamba(num_classes=config['data']['n_classes'])

    if 'swin' in model_type:
        model.head = nn.Linear(model.head.in_features, 8)
        transform = TRANSFORMS['swin']
    elif model_type == 'medmamba':
        transform = TRANSFORMS['medmamba']

    return model, transform


def average_across_gpus(list_, device):
    tensor = torch.tensor(list_).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM) # GLOO doesn't support AVG :(
    tensor /= dist.get_world_size()
    return tensor.tolist()


def train_loop(model, train_loader, val_loader, n_epochs, criterion, optimizer, device, using_dist):
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
            
            predictions = torch.argmax(outputs, dim=1)
            print(outputs)
            print(labels)
            print(predictions)
            loss = criterion(predictions, labels)

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