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


def ensemble_prediction(models, images, num_classes, device):

    # Collect predictions from all models
    predictions = torch.zeros((len(models), num_classes)).to(device)
    for i, model in enumerate(models):
        outputs = model(images)  # Forward pass
        predictions[i] = F.softmax(outputs, dim=1)  # Convert logits to probabilities

    # Average the predictions across models
    return torch.mean(predictions, dim=0)


def evaluate_model(models, test_loader, dataset_name, num_classes, device):
    """
    Evaluate a trained model on a test dataset.

    Args:
        model (torch.nn.Module): Trained model
        test_loader (torch.utils.data.DataLoader): DataLoader for test dataset
        device (torch.device): Device used for inference

    Returns:
        dict: Evaluation metrics for the model
    """

    
    print('Beginning evaluation...')

    all_preds = []
    all_labels = []

    misclassified_bne = []
    misclassified_sne = []

    # Get test set results
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            images, labels = images.to(device), labels.to(device)

            if isinstance(models, list):
                outputs = ensemble_prediction(models, images, num_classes, device)
            else:
                outputs = models(images)

                if not isinstance(models, CompleteClassifier):
                    outputs = torch.argmax(outputs, dim=1)

            if dataset_name == "chula":
                for i in range(images.size(0)):
                    true_label = labels[i].item()
                    predicted_label = outputs[i].item()
                    image_name = image_names[i]

                    # If the image is misclassified and the label is 0 or 3
                    if true_label == 3 and predicted_label == 0:
                        misclassified_bne.append(image_name)
                    if true_label == 0 and predicted_label == 3:
                        misclassified_sne.append(image_name)

            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = get_eval_metrics(all_preds, all_labels)
    if dataset_name == "chula":
        metrics['BNE images misclassified as SNE'] = misclassified_bne
        metrics['SNE images misclassified as BNE'] = misclassified_sne

    # Return evaluation metrics
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

    if isinstance(model_config, list):
        models = []
        for trained_model_path, name in zip(model_config['trained_model_path'], model_config['name']):
            model, transforms = load_model(model_config={'trained_model_path' : trained_model_path, 'name' :  name}, device=device)
            models.append(model, device) # (don't bother storing the transforms for each model, assume all the test transforms are the same)
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
    metrics = evaluate_model(models, test_loader, dataset_config['name'], dataset_config['num_classes'], device)

    # Get runtime
    metrics['Time to evaluate'] = time.time() - start_time

    # Create output directory for log
    date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_{model_config['name']}')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save log
    save_log(out_dir, date, metrics, model_config, dataset_config)
    

if __name__ == "__main__":

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where model evaluation log will be saved (default=cwd)', default='.')
    parser.add_argument('--trained_model_path', nargs='+', type=str, help='Path to trained model .pth', required=True)
    parser.add_argument('--neutrophil_model_path', type=str, help='Path to trained neutrophil model .pth')
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


    if isinstance(trained_model_path, list):
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

    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(out_dir, model_config, batch_size, dataset_config, dataset_download_dir)
