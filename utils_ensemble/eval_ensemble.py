import time
import numpy as np

import torch

from utils.eval import get_eval_metrics


def ensemble_prediction_average(base_models_outputs, device):
    """
    Calculate the average prediction of base models based on average prediction probability.

    Args:
        base_models_outputs (torch.tensor): base model outputs shape (num_models, num_images, num_classes)
        device (torch.device): device to put tensors on

    Returns:
        torch.tensor: final predictions shape (number of images)
    """
  
    # Get all model prediction probabilites for all imags
    probs = torch.zeros(base_model_outputs.size()).to(device) # shape (num_models, num_images, num_classes)
    for i, output in enumerate(base_models_outputs):
        probs[i] = torch.softmax(output, dim=1)

    # Return the class with highest average prediction probability across all models
    average_probs = torch.mean(probs, dim=0)
    return torch.argmax(average_probs, dim=1)


def ensemble_prediction_majority(base_models_outputs, num_images, device):
    """
    Calculate the majority prediction of base models.
    (Uses highest average probability to settle no majority cases e.g. 2 versus 2)

    Args:
        base_models_outputs (torch.tensor): base model outputs shape (number of base models, number of images, number of classes)
        device (torch.device): device to put tensors on

    Returns:
        torch.tensor: final predictions shape (number of images)
    """
    
    # Get all model predictions and probabilites for all images
    preds = torch.zeros(base_model_outputs.size()[:1]).to(device) # shape (num_models, num_images)
    probs = torch.zeros(base_model_outputs.size()).to(device) # shape (num_models, num_images, num_classes)
    for i, output in enumerate(base_models_outputs):
        preds[i] = torch.argmax(output, dim=1)
        probs[i] = torch.softmax(output, dim=1)
    
    # One prediction per image
    out_preds = torch.zeros(base_model_outputs.size(1)).to(device) # shape (num_models, num_images)

    # Get the majority vote for each model
    for i in range(num_images):
        # Get image predictions and probabilities
        image_preds = preds[:, i] # shape (num_models)
        image_probs = probs[:, i, :] # shape (num_models, num_classes)

         # Get all the classes predicted for this model
        image_prediction, counts = image_preds.unique(return_counts=True)

        # Get the class(es) with the highest number of predictions
        majority = counts.max()
        conflicting_classes = image_prediction[counts == majority]

        # If there are multiple classes with highest count, then no majority so resolve
        if len(conflicting_classes) != 1: 

            # Get average probability for all the models that predicted this class
            average_probs = [torch.mean(image_probs[:, int(class_.item())], dim=0) for class_ in conflicting_classes]
            average_probs = torch.stack(average_probs)

            # Set image prediction to the class with the highest probability over all the images
            out_preds[i] = conflicting_classes[torch.argmax(average_probs)]
        else:
            # Otherwise set output prediction to majority vote
            out_preds[i] = image_prediction[counts == majority]

    return out_preds


def ensemble_stacking(base_models_outputs, meta_learner, device):
    """
    Calculate the meta-learner prediction based on base model outputs.

    Args:
        base_models_outputs (torch.tensor): base model outputs shape (number of base models, number of images, number of classes)
        meta_learner (nn.Module): ensemble meta-learner
        device (torch.device): device to put tensors on

    Returns:
        torch.tensor: final predictions shape (number of images)
    """

    # Concatenate base model outputs
    base_models_outputs = torch.cat(base_models_outputs, dim=1) 
    base_models_outputs = base_models_outputs.to(device)

    # Pass the outputs through the stacking model
    meta_learner = meta_learner.to(device)
    outputs = meta_learner(base_models_outputs)

    # Return the meta learner predictions for each image
    return torch.argmax(outputs, dim=1)


def evaluate_model(ensemble_mode, base_models, base_model_order, test_loader, dataset_name, device, meta_learner=None):
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
        for images, labels, _ in test_loader:
            torch.cuda.reset_peak_memory_stats(device)  # Reset memory tracking
            labels = labels.to(device)

            base_models_outputs = []
            for base_model_name, image in zip(base_model_order, images):
                image = image.to(device)

                base_model = base_models[base_model_name]
                base_model.to(device)

                with torch.no_grad():
                    base_model_output = base_model(image)
                base_model.to('cpu')

                base_models_outputs.append(base_model_output.cpu())

            # Get the predictions based on ensemble mode
            if ensemble_mode == 'stacking':
                preds = ensemble_stacking(base_models_outputs, meta_learner, device)

            elif ensemble_mode == 'average':
                preds = ensemble_prediction_average(base_models_outputs, labels.shape[0], device)

            elif ensemble_mode == 'majority':
                preds = ensemble_prediction_majority(base_models_outputs, labels.shape[0], device)

            elif ensemble_mode == 'weighted_average':
                preds = ensemble_prediction_weighted_average(base_models_outputs, labels.shape[0], device)
            
            # Failure cases
            if dataset_name == "chula":
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    predicted_label = outputs[i].item()
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