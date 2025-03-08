import os
from datetime import datetime
import yaml

import torch

from .common import save_log, average_across_gpus


def save_models(out_dir, trained, model_type, epochs):
    """
    Save trained model(s) to a given output directory

    Args:
        out_dir(string): Path to directory to save the model(s) to
        trained(torch.nn.Module or list[torch.nn.Module]): Model or list of models to be saved
        using_dist(bool): Whether multiple GPUs were used to train the model(s)
    """

    # Create trained models directory if needed
    if isinstance(trained, list):
        # Save trained models for each fold
        for idx, model in enumerate(trained):
            model_path = os.path.join(out_dir, f'{model_type}_fold_{idx}_epoch_{epochs}.pth')
            torch.save(model.state_dict(), model_path)
        print(f'\n{len(trained)} trained models saved to {out_dir}')
    else:
        # Save the model
        model_path = os.path.join(out_dir, f'{model_type}_epoch_{epochs}.pth')
        torch.save(trained.state_dict(), model_path)
        print(f'Saved trained model to {model_path}')


def save_config(out_dir, config):
    """
    # Save model config file

    Args:
        out_dir(string): Path to directory to save the config file to
        config(dict): Training config dictionary to be saved
    """

    # Save model configuration
    config_path = os.path.join(out_dir, 'config.yml')
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def save(out_dir, metrics, trained, model_config, dataset_config):
    """
    Save trained models, training log and model configuration file

    Args:
        out_dir(string): Path to directory to save the log, models and config file to
        metrics(dict or list[dict]): Model or list of metrics dictionarys to be saved
        trained(torch.nn.Module or list[torch.nn.Module]): Model or list of models to be saved
        config(dict): Model config dictionary to be saved
        using_dist(bool): Whether multiple GPUs were used to train the model(s)
    """

    # Get date/time of saving
    date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_{model_config['name']}')

    # Create output directory
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Save models, log and config yml
    save_models(out_dir, trained, model_config['name'], model_config['epochs'])
    save_log(out_dir, date, metrics, model_config, dataset_config)
    save_config(out_dir, model_config)


def train_loop(model, model_config, train_loader, val_loader, criterion, optimizer, device, using_dist, out_dir, verbose=False):
    """
    Training loop used for training a single model

    Args:
        model(torch.nn.Module): Model to be trained
        train_loader(torch.utils.data.DataLoader): Training data data loader
        val_loader(torch.utils.data.DataLoader): Validation data data loader
        n_epochs(int): Number of epochs to train for
        criterion(torch.nn.Module): Training loss function
        optimizer(torch.optim): Training optimizer
        device(torch.cuda.device): Id of the device to execute this training loop

    Returns:
        torch.nn.Module: Trained model
        dict: Dictionary containing various training metrics
    """
    train_accuracy_per_epoch = []
    train_loss_per_epoch = []
    val_accuracy_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(model_config['epochs']):
        model.train()

        if using_dist:
            train_loader.sampler.set_epoch(epoch)

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels.squeeze(1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct_train += (preds == labels.squeeze(1)).sum().item()

            total_train += labels.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        best_val_accuracy = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(dim=1))

                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct_val += (preds == labels.squeeze(1)).sum().item()
                total_val += labels.size(0)
                
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

        if val_accuracy > best_val_accuracy and val_accuracy > 0.915:
            best_val_accuracy = val_accuracy
            save_models(out_dir, model, model_config['name'], epoch)

        # Store metrics for this epoch
        train_accuracy_per_epoch.append(train_accuracy)
        train_loss_per_epoch.append(avg_train_loss)
        val_accuracy_per_epoch.append(val_accuracy)
        val_loss_per_epoch.append(avg_val_loss)

        # Print epoch metrics
        if device in [0, 'cuda:0'] and verbose:
            print(f'Epoch [{epoch + 1}/{model_config['epochs']}]:')
            print(f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')


    # Return final metrics
    if using_dist:
        # Average per epoch metrics across over all GPUs
        return model, {
            'Average train Accuracy per epoch': average_across_gpus(train_accuracy_per_epoch, device),
            'Average train Loss per epoch': average_across_gpus(train_loss_per_epoch, device),
            'Average validation Accuracy per epoch': average_across_gpus(val_accuracy_per_epoch, device),
            'Average validation Loss per epoch': average_across_gpus(val_loss_per_epoch, device),
        }
    
    return model, {
        'Train Accuracy per epoch': train_accuracy_per_epoch,
        'Train Loss per epoch': train_loss_per_epoch,
        'Validation Accuracy per epoch': val_accuracy_per_epoch,
        'Validation Loss per epoch': val_loss_per_epoch,
    }