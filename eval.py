import argparse
import yaml
import os
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset, TransformedDataset
from models import init_model
from utils.common import save_log
from utils.eval import get_eval_metrics


torch.backends.cudnn.enabled = True


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


def evaluate_model(models, test_loader, dataset_name, num_classes, device):
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

            if isinstance(models, list):
                outputs = ensemble_prediction_weighted_average(models, images, num_classes, device)
            else:
                outputs = models(images)

                if not isinstance(models, CompleteClassifier):
                    outputs = torch.argmax(outputs, dim=1)

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



# Two-tier model for neutrophils classification
class CompleteClassifier(nn.Module):
    def __init__(self, model_config, dataset_config):
        super().__init__()

        # Load the wbc classifier
        self.wbc_model, self.model_transforms = init_model(model_config, dataset_config['n_classes'])
        self.wbc_model.load_state_dict(torch.load(model_config['trained_model_path'], map_location='cpu'))
        self.wbc_model.eval()

        # Get the indices of the SNE and BNE neutrophils classes
        self.BNE_index = dataset_config['classes'].index('BNE')
        self.SNE_index = dataset_config['classes'].index('SNE')
        self.neutrophils_indices = torch.tensor([self.BNE_index, self.SNE_index])

        # Load the neutrophils classifier
        model_config['trained_model_path'] = model_config['neutrophil_model_path']
        self.neutrophils_model, _ = init_model(model_config, num_classes=2)
        self.neutrophils_model.load_state_dict(torch.load(model_config['neutrophil_model_path'], map_location='cpu'))
        self.neutrophils_model.eval()

    def forward(self, x):
        wbc_out = self.wbc_model(x)
        wbc_type = torch.argmax(wbc_out, dim=1)

         # Mask for images classified as neutrophils
        neutrophil_mask = torch.isin(wbc_type, self.neutrophils_indices.to(wbc_type.device))

        if neutrophil_mask.any():
            # Get indices of images classified as neutrophils
            neutrophil_indices = torch.where(neutrophil_mask)[0]

            # Get neutrophil type prediction
            neutrophil_out = self.neutrophils_model(x[neutrophil_indices])
            neutrophil_type = torch.argmax(neutrophil_out, dim=1)

            # Map the neutrophils binary predictions to WBC classes
            wbc_type[neutrophil_indices] = torch.where(neutrophil_type == 1, self.BNE_index, self.SNE_index)
        
        return wbc_type


def load_model(model_config, device):
    # Load model
    if 'neutrophil_model_path' in model_config: 
        model = CompleteClassifier(model_config, dataset_config)
        transforms = model.model_transforms
        model = model.to(device)
        model.eval()
    else: 
        model, transforms = init_model(model_config, dataset_config['n_classes'], device)
        model.load_state_dict(torch.load(model_config['trained_model_path'], map_location=device))
        model.eval()

    return model, transforms


def main(out_dir, model_config, batch_size, dataset_config, dataset_download_dir):

    # Setup GPU
    device = 'cuda'

    if isinstance(model_config['name'], list):
        models = []
        for trained_model_path, name in zip(model_config['trained_model_path'], model_config['name']):
            model, transforms = load_model(model_config={'trained_model_path' : trained_model_path, 'name' :  name, 'use_improvements' : model_config['use_improvements']}, device=device)
            models.append(model) # (don't bother storing the transforms for each model, assume all the test transforms are the same)
    else:
        models, transforms = load_model(model_config, device)

    # Apply transforms
    test_dataset = get_dataset(dataset_config, dataset_download_dir, test=True)
    test_dataset = TransformedDataset(test_dataset, transforms['test'], test=True)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Track time
    start_time = time.time()

    # Evaluate the model
    metrics = evaluate_model(models, test_loader, dataset_config['name'], dataset_config['n_classes'], device)

    # Get runtime
    metrics['Time to evaluate'] = time.time() - start_time

    # Create output directory for log
    if isinstance(model_config['name'], list):
        date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_ensemble')
    else:
        date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_{model_config['name']}')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save log
    save_log(out_dir, metrics, model_config, dataset_config)
    

if __name__ == "__main__":

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where model evaluation log will be saved (default=cwd)', default='.')
    parser.add_argument('--trained_model_path', nargs='+', type=str, help='Path to trained model .pth', required=True)
    parser.add_argument('--neutrophil_model_path', type=str, help='Path to trained neutrophil model .pth')
    parser.add_argument('--use_improvements', action=argparse.BooleanOptionalAction, help='Whether to use the proposed model improvements.')
    parser.add_argument('--model_type', nargs='+', type=str, help='Model type e.g. "swin", "vmamba" ', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size when evaluating', default=32)
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset .yml used for evaluation', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    trained_model_path = args.trained_model_path
    model_type = args.model_type
    batch_size = args.batch_size
    dataset_config_path= args.dataset_config_path
    dataset_download_dir = args.dataset_download_dir

    if len(trained_model_path) != 1:
        model_config = {
            'trained_model_path' : args.trained_model_path,
            'name' : args.model_type,
        }
    else: 
        model_config = {
            'trained_model_path' : args.trained_model_path[0],
            'name' : args.model_type[0],
        }

    if args.neutrophil_model_path: model_config['neutrophil_model_path'] = args.neutrophil_model_path
    if args.use_improvements: model_config['use_improvements'] = args.use_improvements

    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(out_dir, model_config, batch_size, dataset_config, dataset_download_dir)
