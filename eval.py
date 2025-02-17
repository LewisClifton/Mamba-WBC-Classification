import argparse
import yaml
import os
import datetime as datetime

import torch

from torch.utils.data import DataLoader

from data.dataset import WBC5000dataset, BloodMNIST, TransformedDataset
from models import init_model
from utils.common import save_log
from utils.eval import accuracy, sensitivity, f1_score, confusion_matrix


torch.backends.cudnn.enabled = True


def evaluate_model(config, test_dataset, device):
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

    # Load model
    model, model_transforms = init_model(config['model']['type'], config['data']['num_classes'])

    # Load saved weights
    model.load_state_dict(torch.load(config['model']['path'], map_location=device))
    model.eval()

    # Put on device
    model.to(device)

    # Apply transforms
    test_dataset = TransformedDataset(test_dataset, model_transforms['test'])

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=config['params']['batch_size'])

    preds = []
    labels = []

    # Get test set results
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds.extend(preds.cpu().numpy().tolist())
            labels.extend(labels.cpu().numpy().tolist())

    # Return evaluation metrics
    return {
        "Accuracy" : accuracy(preds, labels),
        "Sensitivity" : sensitivity(preds, labels),
        "F1-Score" : f1_score(preds, labels),
        "Confusion matrix" : confusion_matrix(preds, labels)
    }


def main(out_dir, config):

    # Setup GPU network if required
    device = 'cuda'

    # Train using specified dataset with/without k-fold cross validation
    if config['data']['dataset'] == 'chula':
        # Get dataset
        test_dataset = WBC5000dataset(config['data']['images_dir'], config['data']['labels_path'], wbc_types=config['data']['classes'])

    elif config['data']['dataset'] == 'bloodmnist':
        # Get dataset
        test_dataset = BloodMNIST(split='test', download=True, size=224)

    # Can add more datasets here..
    elif config['data']['dataset'] == 'foo':
        pass

    # Evaluate the model
    metrics = evaluate_model(config, test_dataset, device)

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
    parser.add_argument('--out_dir', type=str, help='Path to directory where trained model and log will be saved (default=cwd)', default='.')
    parser.add_argument('--config_path', type=str, help='Path to model config .yml.', required=True)

    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    config_path = args.config_path
    using_windows = args.using_windows

    # Load config
    with open(config_path, "r") as yml:
        config = yaml.safe_load(yml)

    main(out_dir, config)
