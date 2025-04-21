import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import manifold
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset, TransformedDataset
from models import init_model
from utils.eval import evaluate_model


def extract_features(model, test_loader, device, target_layers=None):
    """
    Get features from the model at the target layer(s).

    Args:
        model_config (dict): model configuration file
        test_loader (torch.utils.data.DataLoader): test data loader
        device (torch.device): torch device to use for inference
        target_layers (list[torch.nn.module]): layers to extract features from
        
    Returns:
        (numpy.ndarray, numpy.ndarray): extracted features and output labels
    """

    activations = {}

    # Register hook in target layers
    def hook_fn(module, input, output):
        activations[module] = output.detach()

    for layer in target_layers:
        layer.register_forward_hook(hook_fn)

    labels, features = [], []

    # Extract features
    for images, target, _ in test_loader:
        images, target = images.to(device), target.numpy().flatten()
        labels.extend(target)

        # Get the activations of the target layers given input images, model output is discarded
        _ = model(images)
        for layer in target_layers:
            f = activations[layer].cpu().numpy()
            f = f.reshape(f.shape[0], -1)
            features.extend(f)

    return np.array(features), np.array(labels)


def plot_tsne(features, labels, dataset_config, model_name=None):
    """
    Evaluate a trained model on a test dataset and report detailed memory metrics.

    Args:
        model_config (dict): model configuration file
        batch_size (int): batch size for data loader
        dataset_config (dict): dataset configuration file
        dataset_download_dir (str): directory to download datasets to if requried
    """
    
    dataset_name = dataset_config['name']

    # Use the correct labels for the dataset
    if dataset_name == 'chula':
        classes = ["SNE", "LYMPH", "MONO", "BNE", "EOSI", "MYBL", "BASO", "MEMY"]
    elif dataset_name == 'bloodmnist':
        classes = ['BASO', 'EOSI', 'ERYTH', 'IMM. GRAN', 'LYMPH', 'MONO', 'NEUT', 'PLT']
    
    # Number of classes
    num_classes = len(classes)

    # Colour map
    viridis_discrete = plt.cm.viridis(np.linspace(0, 1, num_classes))

    # t-SNE
    tsne = manifold.TSNE(n_components=2, perplexity=40.0, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(features)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=ListedColormap(viridis_discrete), alpha=0.7)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, linestyle="--", alpha=0.6)
    class_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=viridis_discrete[i], markersize=10) for i in range(num_classes)]
    ax.legend(class_labels, classes, loc="center right", bbox_to_anchor=(1, 0.5))
    
    # Set output dir
    out_dir = '/user/work/js21767/Project/out/t_sne/'
    os.makedirs(out_dir, exist_ok=True)
    if model_name:
        graph_path = f'/user/work/js21767/Project/out/t_sne/{dataset_name}_{model_name}_tsne.png'
    else:
        graph_path = f'/user/work/js21767/Project/out/t_sne/{dataset_name}_tsne.png'

    # Save plot
    plt.savefig(graph_path, bbox_inches='tight')
    plt.show()
    print(f'Saved graph to {graph_path}')


def main(model_config, batch_size, dataset_config, dataset_download_dir):

    device = 'cuda'

    # Load dataset
    test_dataset = get_dataset(dataset_config, dataset_download_dir, test=True)

    print('Beginning feature extraction for t-SNE...')

    # Load model if given
    model, transforms = init_model(model_config, num_classes, device)
    model.load_state_dict(torch.load(model_config['trained_model_path'], map_location=device))
    model = model.to(device)
    model.eval()

    # Get dataset 
    test_dataset = TransformedDataset(test_dataset, transforms['test'], test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get layers to extract features from
    target_layers = []
    if model_config['name'] == 'mambavision':
        target_layers = [model.model.levels[2].downsample]
    elif model_config['name'] == 'localmamba':
        target_layers = [model.layers[-1].mixer.in_proj]
    elif model_config['name'] == 'swin':
        target_layers = [model.features[-1][-1].norm1]
    elif model_config['name'] == 'vmamba':
        target_layers = [model.layers[-1].blocks[-1].norm2]

    # Extract features
    features, labels = extract_features(model, test_loader, device, target_layers)

    # Plot t-SNE
    print('Plotting graph.')
    plot_tsne(features, labels, dataset_config, model_config['name'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', type=str, help='Path to trained model .pth')
    parser.add_argument('--neutrophil_model_path', type=str, help='Path to trained neutrophil model .pth')
    parser.add_argument('--use_improvements', action=argparse.BooleanOptionalAction, help='Enable model improvements')
    parser.add_argument('--model_type', type=str, help='Model type e.g. "swin", "vmamba"')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--dataset_config_path', type=str, required=True, help='Path to dataset .yml')
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    args = parser.parse_args()

    # Load dataset config
    with open(args.dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    # Prepare model config
    model_config = {
        'trained_model_path': args.trained_model_path,
        'name': args.model_type,
    }

    main(model_config, args.batch_size, dataset_config, args.dataset_download_dir)
