import argparse
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import math

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import init_model


torch.backends.cudnn.enabled = True


def main(model_config, num_classes):

    # Setup GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialise model
    model, transforms_dict = init_model(model_config, num_classes, device)
    transform = transforms_dict["test"]

    # Load model
    model.load_state_dict(torch.load(model_config["trained_model_path"], map_location=device))

    model = model.to(device)

    # Target layers
    if model_config['name'] == 'mambavision':
        target_layers = [model.model.levels[2].downsample]

    if model_config['name'] == 'localmamba':
        target_layers = [model.layers[-1].mixer.in_proj]

    if model_config['name'] == 'swin':
        target_layers = [model.features[-1][-1].norm1]

    if model_config['name'] == 'vmamba':
        target_layers = [model.features[-1][-1].norm1]

    # Data
    images_path = "/tmp/js21767/WBC 5000/"  # Update with your image directory
    labels_path = "/tmp/js21767/labels_test.csv" 

    labels_convert = ['SNE', 'Lymphocyte', 'Monocyte', 'BNE', 'Eosinophil', 'Myeloblast', 'Basophil', 'Metamyelocyte']
    
    labels = pd.read_csv(labels_path)
    labels = labels[labels['label'].isin(labels_convert)]

    sample = labels.sample(n=20)
    images_names = labels['name'].tolist()
    image_labels = labels['label'].tolist() 

    output_dir = "./out"
    os.makedirs(output_dir, exist_ok=True)

    images_processed = 0

    correct = True
    incorrect = True
    count = 0
    for image_name, label in zip(images_names, image_labels):
        image_path = os.path.join(images_path, image_name)
        output_path = os.path.join(output_dir, f"gradcam_{image_name}")

        # Load image
        image = Image.open(image_path)

        input_tensor = transform(image).unsqueeze(0).to(device)

        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)[:, :, ::-1]
        image_normalized = image_array / 255.0
        image = image_normalized.astype(np.float32)


        # Get label
        target = [ClassifierOutputTarget(labels_convert.index(label))]

        def swin_reshape_transform(tensor, height=7, width=7):
            # Reshape the tensor to match a spatial layout (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
            result = tensor.reshape(tensor.size(0), height, width, 768)  # Reshaping to (BATCH, 7, 7, 768)
            
            # Bring channels to the first dimension (CNN style), like (BATCH, CHANNELS, HEIGHT, WIDTH)
            result = result.transpose(2, 3).transpose(1, 2)
            return result


        def localmamba_reshape_transform(tensor):
            """
            Reshape activations from (B, SeqLen, C) to (B, C, H, W)
            where H x W = SeqLen.
            """
            B, SeqLen, C = tensor.shape  # Expecting (B, SeqLen, 384)
            
            # Compute the spatial dimensions
            height = width = int(math.sqrt(SeqLen))
            assert height * width == SeqLen, "SeqLen must be a perfect square!"

            # Reshape into (B, H, W, C)
            result = tensor.view(B, height, width, C)

            # Rearrange to (B, C, H, W)
            result = result.permute(0, 3, 1, 2)  # Move channels to first dimension
            
            return result


        with GradCAM(model=model, target_layers=target_layers, reshape_transform=swin_reshape_transform) as cam:

            grayscale_cam = cam(input_tensor=input_tensor, targets=target)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(image, grayscale_cam, use_rgb=False)

            out_dir = "/user/work/js21767/Project/out/grad_cam/"

            print(f'Target: {target[0].category}. Output:{torch.argmax(cam.outputs, dim=1).cpu().item()}')
            if target[0].category == torch.argmax(cam.outputs, dim=1).cpu().item():
                cv2.imwrite(os.path.join(out_dir, f"{model_config['name']}_{image_name}_correct_grad-cam.jpg"), visualization)
                correct = False
            
            if target[0].category != torch.argmax(cam.outputs, dim=1).cpu().item():
                cv2.imwrite(os.path.join(out_dir, f"{model_config['name']}_{image_name}_incorrect_grad-cam.jpg"), visualization)
                incorrect = False
            count+= 1
            


if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", type=str, help="Path to trained model .pth", required=True)
    parser.add_argument("--model_type", type=str, help='Model type e.g. "swin", "vmamba"', required=True)
    parser.add_argument("--num_classes", type=int, help="Number of dataset classes", required=True)

    # Parse command line args
    args = parser.parse_args()

    model_config = {
        "trained_model_path": args.trained_model_path,
        "name": args.model_type,
    }

    main(model_config, args.num_classes)
