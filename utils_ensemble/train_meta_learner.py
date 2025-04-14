import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def train_loop_meta_learner(meta_learner, base_models, meta_learner_config, train_loader, val_loader, criterion, optimizer, device, verbose=False):
    """
    Training loop used for training a single meta_learner

    Args:
        meta_learner(torch.nn.Module): meta_learner to be trained
        train_loader(torch.utils.data.DataLoader): Training data data loader
        val_loader(torch.utils.data.DataLoader): Validation data data loader
        n_epochs(int): Number of epochs to train for
        criterion(torch.nn.Module): Training loss function
        optimizer(torch.optim): Training optimizer
        device(torch.cuda.device): Id of the device to execute this training loop

    Returns:
        torch.nn.Module: Trained meta_learner
        dict: Dictionary containing various training metrics
    """
    train_accuracy_per_epoch = []
    train_loss_per_epoch = []
    val_accuracy_per_epoch = []
    val_loss_per_epoch = []

    best_val_accuracy = 0
    best_meta_learner = meta_learner

    for epoch in range(meta_learner_config['epochs']):
        meta_learner.train()

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            labels = labels.to(device)

            
            base_models_outputs = []
            for base_model, image in zip(base_models, images):
                image = image.to(device)
                base_model.to(device)

                with torch.no_grad():
                    base_model_output = base_model(image)
                base_model.to('cpu')

                base_models_outputs.append(base_model_output.cpu())

            # For each base model, compute the output for the entire batch of images.
            base_models_outputs = torch.cat(base_models_outputs, dim=1) 

            base_models_outputs = base_models_outputs.to(device)
            meta_learner = meta_learner.to(device)

            # Pass the stacked and flattened outputs to the meta_learner meta-model.
            outputs = meta_learner(base_models_outputs)

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
        meta_learner.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)

                base_models_outputs = []
                for base_model, image in zip(base_models, images):
                    image = image.to(device)
                    base_model.to(device)

                    base_model_output = base_model(image)
                    base_model.to('cpu')

                    base_models_outputs.append(base_model_output)

                # For each base model, compute the output for the entire batch of images.
                base_models_outputs = torch.cat(base_models_outputs, dim=1) 

                base_models_outputs = base_models_outputs.to(device)
                meta_learner = meta_learner.to(device)

                # Pass the stacked and flattened outputs to the meta_learner meta-model.
                outputs = meta_learner(base_models_outputs)

                loss = criterion(outputs, labels.squeeze(dim=1))

                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct_val += (preds == labels.squeeze(1)).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.squeeze(1).cpu().numpy())
        
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

        # Store metrics for this epoch
        train_accuracy_per_epoch.append(train_accuracy)
        train_loss_per_epoch.append(avg_train_loss)
        val_accuracy_per_epoch.append(val_accuracy)
        val_loss_per_epoch.append(avg_val_loss)

        # Print epoch metrics
        if verbose:
            print(f'Epoch [{epoch + 1}/{meta_learner_config['epochs']}]:')
            print(f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Macro accuracy: {macro_accuracy:.4f}')

        if macro_accuracy > best_val_accuracy:
            best_val_accuracy = macro_accuracy
            if macro_accuracy > 91.5:
                best_meta_learner = meta_learner

    
    metrics = {
        'Train Accuracy per epoch': train_accuracy_per_epoch,
        'Train Loss per epoch': train_loss_per_epoch,
        'Validation Accuracy per epoch': val_accuracy_per_epoch,
        'Validation Loss per epoch': val_loss_per_epoch,
        'Best validation accuracy during training' : best_val_accuracy,
    }

    return best_meta_learner, metrics