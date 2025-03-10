import torch
import torch.nn.functional as F

from utils.eval import get_eval_metrics


def ensemble_prediction_weighted_average(models, images, num_classes, device):
    weights = torch.tensor([91.32706374085684, 91.53605015673982, 93.10344827586206, 91.43155694879833, 92.99895506792059], dtype=torch.float32, device=device) # "localmamba mambavision swin vim vmamba"
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Collect predictions from all models
    predictions = torch.zeros((len(models), images.shape[0], num_classes), device=device)
    
    for i, model in enumerate(models):
        outputs = model(images)  # Forward pass
        predictions[i] = F.softmax(outputs, dim=1) * weights[i]  # Apply weight to probabilities

    # Weighted sum of predictions
    weighted_predictions = torch.sum(predictions, dim=0)  # Shape: (batch_size, num_classes)
    
    return torch.argmax(weighted_predictions, dim=1)  # Get final class predictions


def ensemble_prediction_average(models, images, num_classes, device):

    # Collect predictions from all models
    predictions = torch.zeros((len(models), images.shape[0], num_classes)).to(device)
    for i, model in enumerate(models):
        outputs = model(images)  # Forward pass
        predictions[i] = F.softmax(outputs, dim=1)  # Convert logits to probabilities

    # Average the predictions across models
    predictions = torch.mean(predictions, dim=0)
    return torch.argmax(predictions, dim=1)


def ensemble_prediction_majority(models, images, device):

    # Collect predictions from all models
    predictions = torch.zeros((len(models), images.shape[0])).to(device)
    for i, model in enumerate(models):
        outputs = model(images)  # Forward pass
        predictions[i] = torch.argmax(outputs, dim=1)

    # Average the predictions across models
    return torch.mode(predictions, dim=0).values


def evaluate_model(ensemble_mode, base_models, test_loader, dataset_name, device, stacking_model=None):
    """
    Evaluate a trained model on a test dataset and report detailed memory metrics.

    Args:
        models (torch.nn.Module or list): Trained model(s) or ensemble
        test_loader (torch.utils.data.DataLoader): DataLoader for test dataset
        dataset_name (str): Name of the dataset
        num_classes (int): Number of classes in the dataset
        device (torch.device): Device used for inference

    Returns:
        dict: Evaluation metrics including memory usage.
    """

    print('Beginning evaluation...')

    all_preds = []
    all_labels = []

    misclassified_bne = []
    misclassified_sne = []

    max_memory = 0  # Track peak memory usage
    total_memory = 0  # Track cumulative memory usage
    num_batches = 0   # Count number of batches

    # Get test set results
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            torch.cuda.reset_peak_memory_stats(device)  # Reset memory tracking
            images, labels = images.to(device), labels.to(device)

            # Get outputs from the base models
            base_model_outputs = torch.stack([base_model(image) for base_model, image in zip(base_models, images)], dim=1)

            # Get the predictions based on ensemble mode
            if ensemble_mode == 'stacking':
                outputs = stacking_model(base_model_outputs)
                outputs = torch.argmax(outputs, dim=1)

            elif ensemble_mode == 'average':
                outputs = ensemble_prediction_average

            elif ensemble_mode == 'majority':
                outputs = ensemble_prediction_majority

            elif ensemble_mode == 'weighted_average':
                outputs = ensemble_prediction_weighted_average

            if dataset_name == "chula":
                for i in range(images.size(0)):
                    true_label = labels[i].item()
                    predicted_label = outputs[i].item()
                    image_name = image_names[i]

                    if true_label == 3 and predicted_label == 0:
                        misclassified_bne.append(image_name)
                    if true_label == 0 and predicted_label == 3:
                        misclassified_sne.append(image_name)

            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # Track peak and average memory usage
            batch_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # Convert to MB
            max_memory = max(max_memory, batch_memory)  # Store highest memory usage
            total_memory += batch_memory  # Accumulate total memory
            num_batches += 1  # Increment batch count

    # Compute additional memory metrics
    avg_memory = total_memory / num_batches if num_batches > 0 else 0  # Average memory per batch
    num_images = len(all_labels)  # Total test images
    memory_per_image = max_memory / num_images if num_images > 0 else 0  # Memory per image

    metrics = get_eval_metrics(all_preds, all_labels)
    metrics['Peak GPU Memory Usage (MB)'] = max_memory
    metrics['Average GPU Memory Usage per Batch (MB)'] = avg_memory
    metrics['Memory Usage per Image (MB)'] = memory_per_image

    if dataset_name == "chula":
        metrics['BNE images misclassified as SNE'] = misclassified_bne
        metrics['SNE images misclassified as BNE'] = misclassified_sne

    return metrics