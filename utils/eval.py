import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, precision_recall_fscore_support

import torch

from models.complete import CompleteClassifier


def get_eval_metrics(preds, labels):
    """Gets accuracy, precision, sensitivity, F1-score, and confusion matrix."""
    preds = np.array(preds)
    labels = np.array(labels)

    # Accuracy
    correct = (preds == labels.squeeze(1)).sum().item()
    total = labels.size
    accuracy = (correct / total) * 100

    # Precision, recall, F1-score
    precision, sensitivity, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

    # Confusion matrix
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    macro_accuracy = np.nanmean(class_accuracies) * 100  # Handle NaNs if any class has zero samples

    return {
        "Accuracy": accuracy,
        "Macro Accuracy": macro_accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "F1 Score": f1,
        "Confusion Matrix": conf_matrix
    }


def evaluate_model(model, test_loader, dataset_name, device):
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
        for images, labels in test_loader:
            torch.cuda.reset_peak_memory_stats(device)  # Reset memory tracking
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if not isinstance(model, CompleteClassifier):
                outputs = torch.argmax(outputs, dim=1)

            # if dataset_name == "chula":
            #     for i in range(images.size(0)):
            #         true_label = labels[i].item()
            #         predicted_label = outputs[i].item()
            #         image_name = image_names[i]

            #         if true_label == 3 and predicted_label == 0:
            #             misclassified_bne.append(image_name)
            #         if true_label == 0 and predicted_label == 3:
            #             misclassified_sne.append(image_name)

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
