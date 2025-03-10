import torch


def train_loop_ensemble(model, base_models, model_config, train_loader, val_loader, criterion, optimizer, device, verbose=False):
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

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Get outputs from the base models
            base_model_outputs = torch.stack([base_model(image) for base_model, image in zip(base_models, images)], dim=1)

            # Stacking output
            outputs = model(base_model_outputs)
            
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

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Get outputs from the base models
                base_model_outputs = torch.stack([base_model(image) for base_model, image in zip(base_models, images)], dim=1)

                # Stacking output
                outputs = model(base_model_outputs)

                loss = criterion(outputs, labels.squeeze(dim=1))

                val_loss += loss.item()
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
                best_model = model

    
    metrics = {
        'Train Accuracy per epoch': train_accuracy_per_epoch,
        'Train Loss per epoch': train_loss_per_epoch,
        'Validation Accuracy per epoch': val_accuracy_per_epoch,
        'Validation Loss per epoch': val_loss_per_epoch,
        'Best validation accuracy during training' : best_val_accuracy,
    }

    return best_model, metrics