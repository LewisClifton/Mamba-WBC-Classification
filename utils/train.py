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
        model.train()

        if using_dist:
            train_loader.sampler.set_epoch(epoch)

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        counter = 0

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

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
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
                
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if val_accuracy > 91.5:
                #
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