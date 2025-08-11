import time
import numpy as np

import torch
import torch.nn as nn

from utils_ensemble.common import ensemble_prediction_average, ensemble_prediction_majority, ensemble_stacking
from .common import average_across_gpus

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from .common import average_across_gpus


def train_epoch(model, train_loader, kl_criterion, temperature, alpha, criterion, optimiser, device, using_dist=False, verbose=False):
    """
    Training loop used for knowledge distillation

    Args:
        student_model (torch.nn.Module): The student model to be trained
        teacher_model (torch.nn.Module): The teacher model to distill from
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        model_config (dict): Model configuration including epochs, distillation params
        criterion (torch.nn.Module): Loss function for hard labels
        distill_criterion (torch.nn.Module): KLDivLoss for soft labels
        optimiser (torch.optim): Training optimiser
        device (torch.device): Device to train on
        using_dist (bool): Whether using DistributedDataParallel
        verbose (bool): Whether to print verbose output

    Returns:
        torch.nn.Module: Trained student model
        dict: Dictionary containing training and validation metrics
    """
    train_accuracy_per_epoch = []
    train_loss_per_epoch = []

    total_loss = 0
    correct = 0
    total = 0

    for images, labels, image_names, teacher_logits  in train_loader:

        images, labels, teacher_logits = images.to(device), labels.to(device), teacher_logits.to(device)

        student_outputs = model(images)

        loss_soft = kl_criterion(
            nn.functional.log_softmax(student_outputs / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1)
        ) * (temperature ** 2)

        loss_hard = criterion(student_outputs, labels.squeeze(1))

        loss = alpha * loss_soft + (1. - alpha) * loss_hard

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(student_outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = total_loss / total
    train_accuracy = 100. * correct / total

    # Return metrics
    if using_dist:
        metrics = {
            'Average train Accuracy per epoch': average_across_gpus(train_accuracy_per_epoch, device),
            'Average train Loss per epoch': average_across_gpus(train_loss_per_epoch, device),
        }
        return model, metrics

    metrics = {
        'Train Accuracy per epoch': train_accuracy_per_epoch,
        'Train Loss per epoch': train_loss_per_epoch,
    }

    return model, avg_train_loss, train_accuracy


def train_loop(model, model_config, train_loader, kl_criterion, criterion, optimiser, device, using_dist, verbose=False):
    """
    Training loop used for training a single model

    Args:
        model(torch.nn.Module): Model to be trained
        train_loader(torch.utils.data.DataLoader): Training data data loader
        val_loader(torch.utils.data.DataLoader): Validation data data loader
        n_epochs(int): Number of epochs to train for
        criterion(torch.nn.Module): Training loss function
        optimiser(torch.optim): Training optimiser
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

    temperature = model_config.get('distill_temperature', 4.0)
    alpha = model_config.get('distill_alpha', 0.5)

    for epoch in range(model_config['epochs']):

        # Train loop
        model.train()
        if using_dist:
            train_loader.sampler.set_epoch(epoch)
        model, avg_train_loss, train_accuracy = train_epoch(model, train_loader, kl_criterion, temperature, alpha, criterion, optimiser, device, using_dist, verbose)

        train_accuracy_per_epoch.append(train_accuracy)
        train_loss_per_epoch.append(avg_train_loss)

        # Print epoch metrics
        if device in [0, 'cuda:0'] and verbose:
            print(f'Epoch [{epoch + 1}/{model_config['epochs']}]:')
            print(f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}')


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