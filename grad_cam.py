import argparse
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import init_model


torch.backends.cudnn.enabled = True


def save_grad_cam_heatmap(transformed_image, ):
    pass


def main(model_config, num_classes, out_dir):

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

    images_processes 

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

        with GradCAM(model=model, target_layers=target_layers) as cam:

            grayscale_cam = cam(input_tensor=input_tensor, targets=target, aug_smooth=True)

            if target[0].category == torch.argmax(cam.outputs, dim=1).cpu().item(): continue

            grayscale_cam = grayscale_cam[0, :]
            
            visualization = show_cam_on_image(image, grayscale_cam, use_rgb=False)

            print(f'Target: {target[0].category}. Output:{torch.argmax(cam.outputs, dim=1).cpu().item()}')

            cv2.imwrite(os.path.join(out_dir, f"{model_config['name']}_{image_name}_grad-cam.jpg"), visualization)


if __name__ == "__main__":
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="Path to output directory", required=True)
    parser.add_argument("--trained_model_path", type=str, help="Path to trained model .pth", required=True)
    parser.add_argument("--model_type", type=str, help='Model type e.g. "swin", "vmamba"', required=True)
    parser.add_argument("--num_classes", type=int, help="Number of dataset classes", required=True)

    # Parse command line args
    args = parser.parse_args()

    model_config = {
        "trained_model_path": args.trained_model_path,
        "name": args.model_type,
    }

    main(model_config, args.num_classes, args.out_dir)
