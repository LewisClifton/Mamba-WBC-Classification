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
from utils.eval import get_eval_metrics, evaluate_model
from models.complete import CompleteClassifier


torch.backends.cudnn.enabled = True


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


def main(model_config, batch_size, dataset_config, dataset_download_dir):

    # Setup GPU
    device = 'cuda'

    model, transforms = load_model(model_config, device)
    model = model.to(device)

    # Apply transforms
    test_dataset = get_dataset(dataset_config, dataset_download_dir, test=True)
    test_dataset = TransformedDataset(test_dataset, transforms['test'], test=True)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Track time
    start_time = time.time()

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, dataset_config['name'], device)

    # Get runtime
    metrics['Time to evaluate'] = time.time() - start_time

    # Save log
    out_dir = f'{model_config['trained_model_path'].removesuffix(".pth")}_eval'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_log(out_dir, metrics, model_config, dataset_config)
    
if __name__ == "__main__":

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', type=str, help='Path to trained model .pth', required=True)
    parser.add_argument('--neutrophil_model_path', type=str, help='Path to trained neutrophil model .pth')
    parser.add_argument('--use_improvements', action=argparse.BooleanOptionalAction, help='Whether to use the proposed model improvements.')
    parser.add_argument('--model_type', type=str, help='Model type e.g. "swin", "vmamba" ', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size when evaluating', default=32)
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset .yml used for evaluation', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()
    trained_model_path = args.trained_model_path
    model_type = args.model_type
    batch_size = args.batch_size
    dataset_config_path= args.dataset_config_path
    dataset_download_dir = args.dataset_download_dir

    model_config = {
        'trained_model_path' : args.trained_model_path,
        'name' : args.model_type,
    }

    if args.neutrophil_model_path: model_config['neutrophil_model_path'] = args.neutrophil_model_path
    if args.use_improvements: model_config['use_improvements'] = args.use_improvements

    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(model_config, batch_size, dataset_config, dataset_download_dir)
