import argparse
import yaml
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from datasets import get_dataset

def get_class_weights(dataset):

    # Get all the dataset labels
    labels = []    
    for _, label in dataset:
        if hasattr(label, 'item'):
            labels.append(label.item()) # in case dataset returns tensors
        else:
            labels.append(int(label))
    labels = np.array(labels)

    # Calculate class weights
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return weights


if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', type=str, help='Path to dataset config .yml.', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')
   
    # Parse command line args
    args = parser.parse_args()
    dataset_config_path = args.dataset_config_path
    dataset_download_dir = args.dataset_download_dir
    
    with open(dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    if dataset_config['name'] == 'bloodmnist':
        dataset, _ = get_dataset(dataset_config, dataset_download_dir)
    elif dataset_config['name'] == 'chula':
        dataset = get_dataset(dataset_config, dataset_download_dir)

    print(f'Calculating class weights of {dataset_config['name']}.')
    class_weights = get_class_weights(dataset)

    print(f'Class weights: {class_weights}')
    dataset_config['class_weights'] = class_weights.tolist()

    # Save
    with open(args.dataset_config_path, 'w') as yml:
        yaml.dump(dataset_config, yml)

    print('Weights saved back to config.')