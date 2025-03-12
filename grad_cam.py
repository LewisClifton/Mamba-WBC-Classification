import argparse
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from models import init_model


torch.backends.cudnn.enabled = True


def grad_cam(model, image_tensor, target_layer="layer4"):
    """
    Compute Grad-CAM heatmap for an image.
    """
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(next(model.parameters()).device)  # Move to correct device

    # Get feature maps & gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Register hooks to the target layer
    for name, module in model.named_modules():
        if target_layer in name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()  # Predicted class

    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)

    # Compute Grad-CAM
    activations = activations["value"].detach()
    gradients = gradients["value"].detach()
    weights = gradients.mean(dim=[2, 3], keepdim=True)  # Global Average Pooling

    heatmap = (weights * activations).sum(dim=1, keepdim=True)  # Weighted sum
    heatmap = F.relu(heatmap)  # ReLU
    heatmap = heatmap.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap


def save_heatmap(heatmap, original_image, output_path):
    """
    Overlay Grad-CAM heatmap on the original image and save.
    """
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))  # Resize to match image
    heatmap = np.uint8(255 * heatmap)  # Convert to 0-255 scale
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map

    # Convert PIL image to OpenCV format
    image = np.array(original_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # Save image
    cv2.imwrite(output_path, overlay)
    print(f"Saved heatmap to {output_path}")


def main(model_config, num_classes):

    # Setup GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialise model
    model, transforms_dict = init_model(model_config, num_classes, device)
    transform = transforms_dict["test"]

    # Load model
    model.load_state_dict(torch.load(model_config["trained_model_path"], map_location=device))
    model.eval()
    model = model.to(device)

    images_path = ".../"  # Update with your image directory
    images_names = ["foo.png", "bar.png"]

    output_dir = "gradcam_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for image_name in images_names:
        image_path = os.path.join(images_path, image_name)
        output_path = os.path.join(output_dir, f"gradcam_{image_name}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        transformed_image = transform(image)

        # Compute Grad-CAM heatmap
        heatmap = grad_cam(model, transformed_image)

        # Save heatmap overlay
        save_heatmap(heatmap, image, output_path)


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
