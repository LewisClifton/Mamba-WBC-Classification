import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

import torch
import torch.distributed as dist
import torch.nn as nn


def average_across_gpus(list_, device):
    """
    # Average the items in a list across all GPUs used in the process group

    Args:
        list_(list[object]): List of objects to be averaged over each GPU
        device(torch.cuda.device): Id of the device used to call this function

    Returns:
        list[object]: List averaged over each GPU
    """
    tensor = torch.tensor(list_).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM) # GLOO doesn't support AVG :(
    tensor /= dist.get_world_size()
    return tensor.tolist()


def train_epoch(model, train_loader, criterion, optimizer, device):
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss    
        loss = criterion(outputs, labels.squeeze(1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        train_loss += loss.item()
        
        # Track accuracy
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            preds = (torch.sigmoid(outputs) > 0.5).long()  
        else:
            _, preds = outputs.max(1)
    
        correct_train += (preds == labels.squeeze(1)).sum().item()
        total_train += labels.size(0)

    # Calculate training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = (correct_train / total_train) * 100

    return model, avg_train_loss, train_accuracy


def val_epoch(model, val_loader, criterion, device, save_output=False, model_name=None, out_dir=None):
    val_loss = 0.0
    correct_val = 0
    total_val = 0


    all_preds = []
    all_labels = []

    all_image_names = None
    all_outputs = None

    with torch.no_grad():
        for images, labels, image_names in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels.squeeze(dim=1))

            # Track loss
            val_loss += loss.item()

            # Track accuracy
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                preds = (torch.sigmoid(outputs) > 0.5).long()  
            else:
                _, preds = outputs.max(1)

            correct_val += (preds == labels.squeeze(1)).sum().item()
            total_val += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze(1).cpu().numpy())

            if save_output:

                if all_outputs is None:
                    all_outputs = outputs.cpu().numpy()
                else:
                    all_outputs = np.concatenate([all_outputs, outputs.cpu().numpy()], axis=0)

                if all_image_names is None:
                    all_image_names = np.array(image_names)
                else:
                    all_image_names = np.concatenate([all_image_names, image_names], axis=0)


    # Save output if doing stacking ensemble training
    if save_output:

        # Create dataframe with image name and outputs
        outputs_df = pd.DataFrame(all_outputs, columns=[f'{model_name}_{i}' for i in range(all_outputs.shape[1])])
        outputs_df.insert(0, 'name', all_image_names)

        # Get output path
        csv_path = os.path.join(out_dir, f'{model_name}.csv')

        # Check for existing df from previous fold
        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path)
            outputs_df = pd.concat([old_df, outputs_df], ignore_index=True)

        # Save outputs to csv
        outputs_df.to_csv(csv_path, index=False)

        print(f'Saved fold validation outputs to {csv_path}')

        return

    else:

        # Calculate macro accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        conf_matrix = confusion_matrix(all_preds, all_labels)
        class_accuracies = np.where(conf_matrix.sum(axis=1) != 0,
                                    conf_matrix.diagonal() / conf_matrix.sum(axis=1),
                                    0)
        macro_accuracy = np.nanmean(class_accuracies) * 100
                
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

        return val_accuracy, avg_val_loss, macro_accuracy 


def train_loop(model, model_config, train_loader, val_loader, criterion, optimizer, device, using_dist, verbose=False):
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

    best_val_accuracy = 0
    best_model = model

    for epoch in range(model_config['epochs']):

        # Train loop
        model.train()
        if using_dist:
            train_loader.sampler.set_epoch(epoch)
        model, avg_train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)

        train_accuracy_per_epoch.append(train_accuracy)
        train_loss_per_epoch.append(avg_train_loss)

        # Print epoch metrics
        if device in [0, 'cuda:0'] and verbose:
            print(f'Epoch [{epoch + 1}/{model_config['epochs']}]:')
            print(f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}')

        # Validation loop
        if val_loader is not None:
            model.eval()
            val_accuracy, avg_val_loss, macro_accuracy  = val_epoch(model, val_loader, criterion, device)

            val_accuracy_per_epoch.append(val_accuracy)
            val_loss_per_epoch.append(avg_val_loss)

            # Print epoch metrics
            if device in [0, 'cuda:0'] and verbose:
                print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Macro accuracy: {macro_accuracy:.4f}')

            if macro_accuracy > best_val_accuracy:
                best_val_accuracy = macro_accuracy
                if macro_accuracy > 91.5:
                    best_model = model
        

    # Return final metrics
    if using_dist:

        metrics = {
            'Average train Accuracy per epoch': average_across_gpus(train_accuracy_per_epoch, device),
            'Average train Loss per epoch': average_across_gpus(train_loss_per_epoch, device),
            'Average validation Accuracy per epoch': average_across_gpus(val_accuracy_per_epoch, device),
            'Average validation Loss per epoch': average_across_gpus(val_loss_per_epoch, device),
            'Best validation accuracy during training' : average_across_gpus(best_val_accuracy),
        }

        # Average per epoch metrics across over all GPUs
        return best_model, metrics
    
    metrics = {
        'Train Accuracy per epoch': train_accuracy_per_epoch,
        'Train Loss per epoch': train_loss_per_epoch,
        'Validation Accuracy per epoch': val_accuracy_per_epoch,
        'Validation Loss per epoch': val_loss_per_epoch,
        'Best validation accuracy during training' : best_val_accuracy,
    }

    return best_model, metrics