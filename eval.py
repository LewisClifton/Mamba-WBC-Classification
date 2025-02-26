import argparse
import yaml
import os
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, TransformedDataset
from models import init_model
from utils.common import save_log
from utils.eval import get_eval_metrics


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
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Return evaluation metrics
    return get_eval_metrics(all_preds, all_labels)


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

    # Track time
    start_time = time.time()

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, device)

    # Get runtime
    metrics['Time to evaluate'] = time.time() - start_time

    # Create output directory for log
    date = datetime.now().strftime('%Y_%m_%d_%p%I_%M')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save log
    save_log(out_dir, date, metrics, model_config, dataset_config)
    

if __name__ == "__main__":

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where model evaluation log will be saved (default=cwd)', default='.')
    parser.add_argument('--trained_model_path', type=str, help='Path to trained model .pth', required=True)
    parser.add_argument('--model_type', type=str, help='Model type e.g. "swin", "vmamba" ', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size when evaluating', default=32)
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset .yml used for evaluation', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    dataset_config_path= args.dataset_config_path
    dataset_download_dir = args.dataset_download_dir

    # Get dataset configs
    model_config = {
        'trained_model_path' : args.trained_model_path,
        'name' : args.model_type,
        'batch_size' : args.batch_size,
    }
    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(out_dir, model_config, dataset_config, dataset_download_dir)
