import argparse
import yaml
import os
import datetime as datetime

import torch

from torch.utils.data import DataLoader

from datasets import get_dataset, TransformedDataset
from models import init_model
from utils.common import save_log
from utils.eval import accuracy, sensitivity, f1_score, confusion_matrix


torch.backends.cudnn.enabled = True


def evaluate_model(model, test_loader, device):
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

    # Get test set results
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Return evaluation metrics
    return {
        "Accuracy" : accuracy(all_preds, all_labels),
        "Sensitivity" : sensitivity(all_preds, all_labels),
        "F1-Score" : f1_score(all_preds, all_labels),
        "Confusion matrix" : confusion_matrix(all_preds, all_labels)
    }


def main(out_dir, model_config, dataset_config, dataset_download_dir):

    # Setup GPU
    device = 'cuda'

    # Load model
    model, model_transforms = init_model(model_config, dataset_config['n_classes'])

    # Load saved weights
    model.load_state_dict(torch.load(model_config['trained_model_path'], map_location=device))
    model.eval()

    # Put on device
    model.to(device)

    # Apply transforms
    test_dataset = get_dataset(dataset_config, dataset_download_dir, test=True)
    test_dataset = TransformedDataset(test_dataset, model_transforms['test'])

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'])

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, device)

    # Create output directory for log
    date = datetime.now().strftime('%Y_%m_%d_%p%I_%M')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save log
    save_log(out_dir, date, metrics)
    

if __name__ == "__main__":

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where model evaluation log will be saved (default=cwd)', default='.')
    parser.add_argument('--model_config_path', type=str, help='Path to model config .yml.', required=True)
    parser.add_argument('--dataset_config_path', type=str, help='Name of dataset to evaluate model with', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    model_config_path = args.model_config_path
    dataset_config_path= args.dataset_config_path
    dataset_download_dir = args.dataset_download_dir

    # Get the model and dataset configs
    with open(model_config_path, 'r') as yml:
        model_config = yaml.safe_load(yml)
    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(out_dir, model_config, dataset_config, dataset_download_dir)
