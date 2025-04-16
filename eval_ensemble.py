import argparse
import yaml
import os
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, EnsembleDataset, TransformedDataset
from utils.common import save_log

from utils_ensemble.common import get_ensemble
from utils_ensemble.eval_ensemble import evaluate_model


torch.backends.cudnn.enabled = True


def main(out_dir, ensemble_config, dataset_config, dataset_download_dir):

    # Setup GPU
    device = 'cuda'

    # Initialise model
    stacking_model, base_models, base_models_transforms = get_ensemble(ensemble_config, dataset_config['n_classes'], device)

    # Ensures data is fed to models in a consistent order
    if 'base_model_order' in ensemble_config:
        base_model_order = ensemble_config['base_model_order']
    else:
        base_model_order = base_models.keys()

    # Initialise dataset
    test_dataset = get_dataset(dataset_config, dataset_download_dir, test=True)
    test_dataset = EnsembleDataset(test_dataset, [base_models_transforms[base_model]['test'] for base_model in base_model_order], test=True)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=ensemble_config['batch_size'], shuffle=False, num_workers=1)

    # Evaluate the model
    metrics = evaluate_model(ensemble_config['ensemble_mode'], base_models, base_model_order, test_loader, dataset_config['name'], device, stacking_model=stacking_model)

    # Create output directory for log
    date = datetime.now().strftime(f'%Y_%m_%d_%p%I_%M_ensemble')
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save log
    save_log(out_dir, metrics, ensemble_config, dataset_config)
    

if __name__ == "__main__":

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where model evaluation log will be saved (default=cwd)', default='.')
    parser.add_argument('--ensemble_config_path', type=str, help='Path to ensemble config .yml', required=True)
    parser.add_argument('--meta_learner_path', type=str, help='Path to the trained stacking ensemble .pth to be evaluated')
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset .yml used for evaluation', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    ensemble_config_path = args.ensemble_config_path
    dataset_config_path= args.dataset_config_path
    dataset_download_dir = args.dataset_download_dir

    with open(ensemble_config_path, 'r') as yml:
        ensemble_config = yaml.safe_load(yml)
    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    if ensemble_config['ensemble_mode'] == 'stacking':
        ensemble_config['meta_learner_path'] = args.meta_learner_path

    main(out_dir, ensemble_config, dataset_config, dataset_download_dir)
