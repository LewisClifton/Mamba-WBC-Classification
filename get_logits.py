import argparse
import os
import numpy as np
from PIL import Image
import pandas as pd
import math
import yaml
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from models import init_model
from dataset import get_dataset, TransformedDataset
from utils_ensemble.common import get_ensemble, ensemble_stacking, ensemble_prediction_average, ensemble_prediction_majority


torch.backends.cudnn.enabled = True


def get_model_outputs(model, test_loader, device):
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

    all_image_names = []
    all_preds = []

    # Get test set results
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(images)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_image_names.extend(list(image_names))

    return all_preds, all_image_names


def get_ensemble_outputs(base_models, model_config, test_loader, device, meta_learner=None):
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

    all_image_names = []
    all_preds = []

    # Get test set results
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            torch.cuda.reset_peak_memory_stats(device)  # Reset memory tracking
            labels = labels.to(device)

            base_models_outputs = []
            for base_model_name, image in zip(model_config['base_models_order'], images):
                image = image.to(device)

                base_model = base_models[base_model_name]
                base_model.to(device)

                with torch.no_grad():
                    base_model_output = base_model(image)
                base_model.to('cpu')

                base_models_outputs.append(base_model_output.cpu())

            # Get the predictions based on ensemble mode
            if model_config['ensemble_mode'] == 'stacking':
                preds = ensemble_stacking(base_models_outputs, meta_learner, device)

            elif model_config['ensemble_mode'] == 'average':
                preds = ensemble_prediction_average(base_models_outputs, labels.shape[0], device)

            elif model_config['ensemble_mode'] == 'majority':
                preds = ensemble_prediction_majority(base_models_outputs, labels.shape[0], device)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_image_names.extend(list(image_names))

    return all_preds, all_image_names


def main(out_dir, model_config, dataset_config, dataset_download_dir):

    # Setup GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get dataloader
    dataset = get_dataset(dataset_config, dataset_download_dir, test=True)
    dataset = TransformedDataset(dataset)
    train_loader = DataLoader(dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=1, drop_last=False)

    print(f'Producing logits for {model_config['name']} for the {dataset_config['name']} dataset...')

    # Initialise teacher model
    if 'base_models' not in model_config:
        model = init_model(model_config, dataset_config['num_classes'], device)
        model = model.to(device)   

        all_preds, all_image_names = get_model_outputs(model, train_loader, device)

    else:
        # Ensemble if required
        meta_learner, ensemble, _ = get_ensemble(model_config, dataset_config['num_classes'], device)
        ensemble = ensemble.to(device)   

        all_preds, all_image_names = get_ensemble_outputs(ensemble, model_config, train_loader, device, meta_learner=meta_learner)

    # Logits columns
    df_logits = pd.DataFrame(all_preds)
    num_classes = dataset_config['num_classes']
    df_logits.columns = [f'logit_{i}' for i in range(num_classes)]

    # Combine with image names
    df = pd.DataFrame({'name': all_image_names})
    df = pd.concat([df, df_logits], axis=1)

    # Set up output directory
    date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_{model_config['name']}')
    out_dir = os.path.join(args.out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save
    print(f'Saving logits to {out_dir}')
    df.to_csv(os.path.join(out_dir,  'logits.csv'))


if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where Grad-CAM heatmaps are saved.', required=True)
    parser.add_argument('--model_config_path', type=str, help='Path to model config .yml.', required=True)
    parser.add_argument("--trained_model_path", type=str, help="Path to trained model .pth (if not using ensemble)", required=True)
    parser.add_argument("--dataset_config_path", type=str, help="Path to trained model .pth", required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()

    # Get the model and dataset configs
    with open(args.model_config_path, 'r') as yml:
        model_config = yaml.safe_load(yml)

    with open(args.dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(args.out_dir, model_config, dataset_config, args.dataset_download_dir)
