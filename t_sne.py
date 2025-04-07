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
from models.complete import CompleteClassifier
from utils.eval import evaluate_model


def load_model(model_config, dataset_config, device):
    """ Load the model and apply necessary transformations. """
    if 'trained_model_path' in model_config:
        if 'neutrophil_model_path' in model_config:
            model = CompleteClassifier(model_config, dataset_config)
            transforms = model.model_transforms
        else:
            model, transforms = init_model(model_config, dataset_config['n_classes'], device)
            model.load_state_dict(torch.load(model_config['trained_model_path'], map_location=device))

        model = model.to(device)
        model.eval()
        
        return model, transforms
    else:
        return None, None


def extract_features(model, test_loader, device, target_layers=None):
    """ Extract features from a specified layer using hooks. """
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output.detach()

    # Register hooks for each target layer if model exists
    if model:
        for layer in target_layers:
            layer.register_forward_hook(hook_fn)

    labels, features = [], []

    # Extract features
    for images, target in test_loader:
        images, target = images.to(device), target.numpy().flatten()
        labels.extend(target)

        with torch.no_grad():
            if model:
                _ = model(images)  # Forward pass triggers hook
                for layer in target_layers:
                    f = activations[layer].cpu().numpy()
                    f = f.reshape(f.shape[0], -1)  # Flatten feature maps
                    features.extend(f)
            else:
                # No model case: directly flatten images as features
                features.extend(images.cpu().numpy().reshape(images.shape[0], -1))

    return np.array(features), np.array(labels)


def plot_tsne(features, labels, dataset_config, model_name=None):
    """ Plot t-SNE visualization of extracted features with discrete colors for classes and legend. """
    
    dataset_name = dataset_config['name']

    if dataset_name == 'chula':
        classes = ["SNE", "LYMPH", "MONO", "BNE", "EOSI", "MYBL", "BASO", "MEMY"]
    elif dataset_name == 'bloodmnist':
        classes = ['BASO', 'EOSI', 'ERYTH', 'IMM. GRAN', 'LYMPH', 'MONO', 'NEUT', 'PLT']
    
    # Number of classes
    n_classes = len(classes)

    # Create a discrete version of the viridis colormap
    viridis = plt.cm.viridis  # Continuous colormap

    import numpy as np
    viridis_discrete = viridis(np.linspace(0, 1, n_classes))  # Discretize it

    # t-SNE transformation
    tsne = manifold.TSNE(n_components=2, perplexity=40.0, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(features)

    # Adjust figure size to fit the legend outside
    fig, ax = plt.subplots(figsize=(8, 4))

    # Scatter plot with discrete colors for each class
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=ListedColormap(viridis_discrete), alpha=0.7)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=viridis_discrete[i], markersize=10) for i in range(n_classes)]
    
    # Place the legend outside the plot area
    ax.legend(handles, classes, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))

    ax.grid(True, linestyle="--", alpha=0.6)

    out_dir = '/user/work/js21767/Project/out/t_sne/'
    os.makedirs(out_dir, exist_ok=True)

    if model_name:
        graph_path = f'/user/work/js21767/Project/out/t_sne/{dataset_name}_{model_name}_tsne.png'
    else:
        graph_path = f'/user/work/js21767/Project/out/t_sne/{dataset_name}_tsne.png'

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    plt.savefig(graph_path, bbox_inches='tight')  # Ensure legend is fully visible
    plt.show()
    print(f'Saved graph to {graph_path}')


def main(model_config, batch_size, dataset_config, dataset_download_dir):
    """ Main function to load model, extract features, and plot t-SNE. """
    device = 'cuda'

    # Load dataset
    test_dataset = get_dataset(dataset_config, dataset_download_dir, test=True)

    print('Beginning feature extraction for t-SNE...')
    if model_config['name'] is not None:
        # Load model if model path is provided, otherwise use raw dataset features
        model, transforms = load_model(model_config, dataset_config, device)

        test_dataset = TransformedDataset(test_dataset, transforms['test'], test=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Determine target layers based on model type (only if model is provided)
        target_layers = []

        if model_config['name'] == 'mambavision':
            target_layers = [model.model.levels[2].downsample]  # Specify the layer for mambavision
        elif model_config['name'] == 'localmamba':
            target_layers = [model.layers[-1].mixer.in_proj]  # Last convolution layer for localmamba
        elif model_config['name'] == 'swin':
            target_layers = [model.features[-1][-1].norm1]  # Last normalization layer for swin
        elif model_config['name'] == 'vmamba':
            target_layers = [model.layers[-1].blocks[-1].norm2]

        # Extract features
        features, labels = extract_features(model, test_loader, device, target_layers)

    else:
        # If no model is passed, extract images and labels from the original dataset
        features, labels = [], []

        for image, target in test_dataset:
            # Convert PIL image to NumPy array
            image_array = np.array(image)  # Convert image to NumPy array
            image_array = image_array.flatten()  # Flatten to 1D vector
            
            features.append(image_array)
            labels.append(target)

        # Convert lists to NumPy arrays
        features = np.array(features)
        labels = np.array(labels)

        # **Ensure features are 2D**
        features = features.reshape(features.shape[0], -1)  # Reshape to (num_samples, num_features)


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
    if args.neutrophil_model_path:
        model_config['neutrophil_model_path'] = args.neutrophil_model_path
    if args.use_improvements:
        model_config['use_improvements'] = args.use_improvements

    main(model_config, args.batch_size, dataset_config, args.dataset_download_dir)
