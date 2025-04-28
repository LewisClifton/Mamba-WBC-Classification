import argparse
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import math
import yaml

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from models import init_model
from datasets import get_dataset, TransformedDataset


torch.backends.cudnn.enabled = True

# transforms based on https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
def swin_reshape_transform(tensor):
    result = tensor.reshape(tensor.size(0), 7, 7, 768)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def localmamba_reshape_transform(tensor):
    result = tensor.reshape(tensor.size(0), 14, 14, 768)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# TODO: add the layers for the other models

def get_grad_cam_config(model_name, model):
    """
    Get the target layers and feature reshape transformation for a given model

    Args:
        model_name (string): Model name
        model (nn.Module): Model

    Return:
        list[nn.Module]: List of layers for Grad-CAM feature extraction
        func(tensor -> tensor): Reshape transformation for layer features 
    """

    
    if model_name == 'swin':
        target_layer = [model.features[-1][-1].norm1]
        reshape_transform = swin_reshape_transform

    elif model_name == 'mambavision':
        target_layer = [model.model.levels[2].downsample]
        reshape_transform = None

    elif model_name == 'localmamba':
        target_layer = [model.layers[-1].mixer.in_proj]
        reshape_transform = localmamba_reshape_transform

    elif model_name == 'vmamba':
        target_layer = [model.layers[-2].downsample[3]]
        reshape_transform = None

    elif model_name == 'vim':
        target_layer = None
        reshape_transform = None

    elif model_name == 'medmamba':
        target_layer = None
        reshape_transform = None

    return target_layer, reshape_transform


def save_log(out_dir, correct_image_names, incorrect_image_names):
    """
    Save training log

    Args:
        out_dir (string): Path to directory to save the log file to
        correct_image_names (list[string]): List of image names that were predicted correctly
        incorrect_image_names (list[string]): List of image names that were predicted correctly
    """

    log_path = os.path.join(out_dir, 'log.txt')
    with open(log_path , 'w') as file:
        file.write(f'Correct images:\n')
        file.write(str(correct_image_names))

        file.write(f'\nIncorrect images:\n')
        file.write(str(incorrect_image_names))

    print(f'\nSaved log to {log_path}')


def grad_cam(out_dir, model, dataset, target_layers, reshape_transform, device):

    # Get prediction
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    correct_image_names = []
    incorrect_image_names = []

    # Iterate through dataset
    counter = 0
    for image, label, image_name in dataset:

        # Convert image
        image = np.array(image.resize((224, 224)))

        # Get model output                        
        image_transformed = transform(image).unsqueeze(0)
        image_transformed = image_transformed.to(device)

        image = np.float32(image) / 255     

        # Get label
        target = [ClassifierOutputTarget(int(label.item()))]

        # Setup Grad-CAM feature tracking
        with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:

            # Get the feature activations at the given layer for the image label
            grayscale_cam = cam(input_tensor=image_transformed, targets=target)[0, :]

            # Put the Grad-CAM heatmap on the image
            grad_cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=False)

            # Check if the prediction is correct
            is_correct = target[0].category == torch.argmax(cam.outputs, dim=1).cpu().item()

            # Get heatmap image output path
            image_name_string = image_name.removesuffix('jpg')
            if is_correct:
                correct_image_names.append(image_name)
                image_name = f"correct/{model_config['name']}_{image_name_string}_correct_grad_cam.jpg"
            else:
                incorrect_image_names.append(image_name)
                image_name = f'incorrect/{model_config['name']}_{image_name_string}_incorrect_grad_cam.jpg' 
                
            
            # Save heatmap image
            image_path = os.path.join(out_dir, image_name)
            cv2.imwrite(image_path, grad_cam_image)
            print(f'Saved heatmap image to {image_path}.')

        counter += 1

        if counter > 5:
            #quit()
            pass

    # Save the correct and incorrect image names
    save_log(out_dir, correct_image_names, incorrect_image_names)


def main(out_dir, model_config, dataset_config, dataset_download_dir):

    # Setup GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialise model
    model, _ = init_model(model_config, dataset_config['num_classes'], device)
    model.eval()

    # Get dataloader
    dataset = get_dataset(dataset_config, dataset_download_dir, test=True)

    # Grad-CAM layers and feature transformation
    target_layers, reshape_transform = get_grad_cam_config(model_config['name'], model)

    # Do Grad-CAM
    print(f'Beginning Grad-CAM for {model_config["name"]} on the {dataset_config["name"]} dataset.')
    grad_cam(out_dir, model, dataset, target_layers, reshape_transform, device)


if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to directory where Grad-CAM heatmaps are saved.', required=True)
    parser.add_argument("--trained_model_path", type=str, help="Path to trained model .pth", required=True)
    parser.add_argument("--dataset_config_path", type=str, help="Path to trained model .pth", required=True)
    parser.add_argument("--model_type", type=str, help='Model type e.g. "swin", "vmamba"', required=True)
    parser.add_argument('--dataset_download_dir', type=str, help='Directory to download dataset to')

    # Parse command line args
    args = parser.parse_args()

    model_config = {
        "pretrained_model_path": args.trained_model_path,
        "name": args.model_type,
    }
    
    # Make output directory
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'incorrect'), exist_ok=True)

    # Get the model and dataset configs
    with open(args.dataset_config_path, 'r') as yml:
        dataset_config = yaml.safe_load(yml)

    main(args.out_dir, model_config, dataset_config, args.dataset_download_dir)
