import argparse
import yaml
import time
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from datasets import get_dataset, EnsembleDataset
from utils.common import save_models, save
from utils_ensemble.common import get_ensemble
from utils_ensemble.train_ensemble import train_loop_ensemble

torch.backends.cudnn.enabled = True


def train_Kfolds(num_folds, ensemble_config, dataset_config, dataset, device, out_dir, verbose=False):
    """
    Train a ensemble with 5-fold cross-validation

    Args:
        config (dict): Dictionary containing training configuration with top level keys of "model","data","params"
        dataset (torch.utils.data.Dataset): Dataset used for training
        device (torch.cuda.device): Device used for training

    Returns:
        list[torch.Module]: List of trained models, one for each fold
        list[dict]: List of training metrics for each of the models
    """
    print(f'\nTraining ensemble for 5 folds.')

    # List of 5 dicts (1 for each fold), containing metrics for each fold
    all_metrics = []

    # Trained models for each fold
    all_trained = []

    # 5-Fold cross-validation setup
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(folds.split(dataset)):
        if verbose:
            print(f'\nFold {fold + 1}/{num_folds}:')
        
        # Create train and validation subsets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        # Train this fold
        trained, metrics = train_ensemble(ensemble_config, dataset_config, train_dataset, val_dataset, device, out_dir, verbose)
        save_models(out_dir, trained, 'ensemble', metrics, fold=fold+1)
    
        # Aggregate models and metrics from this fold
        all_trained.append(trained)
        all_metrics.append(metrics)

    return all_trained, all_metrics


def train_ensemble(ensemble_config, dataset_config, train_dataset, val_dataset, device, out_dir, verbose=False):
    """
    Train a single model

    Args:
        config(dict): Dictionary containing training configuration with top level keys of "model","data","params"
        train_dataset(torch.utils.data.Dataset): Dataset used for training
        val_dataset(torch.utils.data.Dataset): Dataset used for validation
        device(torch.cuda.device): Device used for training

    Returns:
        torch.Module: Trained model
        dict: Training metrics for the model
    """

    print('Training...')

    # Initialise model
    ensemble, base_models, base_models_transforms = get_ensemble(ensemble_config, dataset_config['n_classes'], device)

    # Initialise data loaders
    train_dataset = EnsembleDataset(train_dataset, [transform['train'] for transform in base_models_transforms])
    train_loader = DataLoader(train_dataset, batch_size=ensemble_config['batch_size'], shuffle=True, num_workers=1)
    
    val_dataset = EnsembleDataset(val_dataset, [transform['test'] for transform in base_models_transforms])
    val_loader = DataLoader(val_dataset, batch_size=ensemble_config['batch_size'], shuffle=False, num_workers=1)

    # Create criterion
    if 'class_weights' in ensemble_config.keys():
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(ensemble_config['class_weights']))
    else:
        criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = optim.AdamW(ensemble.parameters(), lr=ensemble_config['learning_rate'], weight_decay=ensemble_config['optim_weight_decay'])

    start_time = time.time()

    # Train the ensemble
    trained, metrics = train_loop_ensemble(ensemble, base_models, ensemble_config, train_loader, val_loader, criterion, optimizer, device, verbose)

    if verbose:
        print('Done.')

    # Get runtime
    metrics['Time to train'] = time.time() - start_time

    return trained, metrics
    

def main(device, out_dir, ensemble, dataset_config, num_folds, dataset_download_dir, verbose=False):

    # Train using specified dataset with/without k-fold cross validation
    if dataset_config['name'] == 'chula':
        # Get dataset
        dataset = get_dataset(dataset_config, dataset_download_dir)

        # Train the ensemble using k-fold cross validation and get the training metrics for each fold
        trained, metrics = train_Kfolds(num_folds, ensemble, dataset_config, dataset, device, out_dir, verbose)

    elif dataset_config['name'] == 'bloodmnist':
        # Get dataset
        train_dataset, val_dataset = get_dataset(dataset_config, dataset_download_dir)

        # Train ensemble only once (i.e. without k-fold cross validation)
        trained, metrics = train_ensemble(ensemble, dataset_config, train_dataset, val_dataset, device, out_dir, verbose)
        save_models(out_dir, trained, 'ensemble', metrics)

    # Can add more datasets here..
    elif dataset_config['name'] == 'foo':
        pass

    save(out_dir, metrics, trained, ensemble, dataset_config)
        

if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where trained ensemble and log will be saved (default=cwd)', default='.')
    parser.add_argument('--ensemble_config_path', type=str, help='Path to ensemble config .yml.', required=True)
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset config .yml.', required=True)
    parser.add_argument('--num_folds', type=int, help='Number of folds for cross fold validation if desired. (default=1)', default=1)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Whether to print per epoch metrics during training')
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')
   
    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    ensemble_config_path = args.ensemble_config_path
    dataset_config_path = args.dataset_config_path
    num_folds = args.num_folds
    verbose = args.verbose
    dataset_download_dir = args.dataset_download_dir

    # Get the ensemble and dataset configs
    with open(ensemble_config_path, 'r') as yml:
        ensemble = yaml.safe_load(yml)
    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_ensemble')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = 'cuda'

    # Create process group if using multi gpus on Linux
    main(device, out_dir, ensemble, dataset_config, num_folds, dataset_download_dir, verbose)