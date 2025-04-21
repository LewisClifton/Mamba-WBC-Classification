import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time

import torch


def get_eval_metrics(preds, labels):
    """
    Load model if one is provided.

    Args:
        preds (numpy.ndarray): Model predictions
        labels (numpy.ndarray): True labels

    Returns:
        dict: Evaluation metric scores
    """

    preds = np.array(preds)
    labels = np.array(labels)

    # Accuracy
    correct = (preds == labels.squeeze(1)).sum().item()
    total = labels.size
    accuracy = (correct / total) * 100

    # (class-weighted) Precision, recall, F1-score
    precision, sensitivity, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, preds)

    # Compute per-class accuracy and macro accuracy
    class_accuracies = np.where(conf_matrix.sum(axis=1) != 0,
                                conf_matrix.diagonal() / conf_matrix.sum(axis=1),
                                0)
    macro_accuracy = np.nanmean(class_accuracies) * 100  # Fair metric across classes


    return {
        "Accuracy": accuracy,
        "Macro Accuracy": macro_accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Sensitivity": sensitivity,
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

    # Aggregate all predictions and labels for evaluation
    all_preds = []
    all_labels = []
    
    # Track failure cases
    misclassified_bne = []
    misclassified_sne = []

    # Track memory usage
    max_memory = 0 
    total_memory = 0
    num_batches = 0

    # Track time
    start_time = time.time()
    
    # Get test set results
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            torch.cuda.reset_peak_memory_stats(device)  # Reset memory tracking
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Failure case
            if dataset_name == "chula":
                for i in range(images.size(0)):
                    true_label = labels[i].item()
                    predicted_label = preds[i].item()
                    image_name = image_names[i]

                    # if BNE but predicted SNE
                    if true_label == 3 and predicted_label == 0:
                        misclassified_bne.append(image_name)

                    # if SNE but predicted BNE
                    if true_label == 0 and predicted_label == 3:
                        misclassified_sne.append(image_name)

            # Aggregate predictions and labels for evaluation
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # Track memory usage
            batch_memory = torch.cuda.max_memory_allocated(device) / 1024**2 
            max_memory = max(max_memory, batch_memory)  
            total_memory += batch_memory
            num_batches += 1 

    
    # Performance metrics
    metrics = get_eval_metrics(all_preds, all_labels)

    # Calculate memory usage metrics
    avg_memory = total_memory / num_batches if num_batches > 0 else 0  # Average memory per batch
    num_images = len(all_labels)  # Total test images
    memory_per_image = max_memory / num_images if num_images > 0 else 0  # Memory per image

    # Add computational efficiency metrics
    runtime = time.time() - start_time
    metrics['Time to evaluate'] = runtime
    metrics['Throughput'] = num_images / runtime
    metrics['Peak GPU Memory Usage (MB)'] = max_memory
    metrics['Average GPU Memory Usage per Batch (MB)'] = avg_memory
    metrics['Memory Usage per Image (MB)'] = memory_per_image

    # Failure cases
    if dataset_name == "chula":
        metrics['BNE images misclassified as SNE'] = misclassified_bne
        metrics['SNE images misclassified as BNE'] = misclassified_sne

    return metrics
