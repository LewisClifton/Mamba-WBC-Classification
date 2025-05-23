import pandas as pd
import os
import torchvision.transforms as transforms
from PIL import Image
import random
import os
import argparse
import numpy as np

from medmnist import BloodMNIST

# (Execute this script from project root)

def augment(out_dir):
    """
    Augment the dataset and save a labels.csv containing the labels for generated images. This will be combined with the existing train dataset
    during training

    Args:
        out_dir (string): Path to directory to save augmented images to
    """
    
    # Augmentation transform
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=360,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),                
    ])

    # Get dataset
    dataset = BloodMNIST(split='train', download=True, size=224, root='/user/work/js21767/')
    
    # Get the class frequencies
    value_counts = np.unique(dataset.labels[:, 0], return_counts=True)

    # Create output dir
    aug_dir = os.path.join(out_dir, 'aug')
    os.makedirs(aug_dir, exist_ok=True)

    augmented_rows = []
    counter = 0

    # Balance dataset
    for label, count in value_counts.items():
        diff = 2000 - count
        if diff <= 0:
            continue 

        # Get all images of this label
        label_indices = np.where(dataset.labels[:, 0] == label)[0]
        
        for i in range(diff):
            # Select a random image from the existing class
            img_idx = random.choice(label_indices)
            image = dataset.imgs[img_idx]  # Fetch the image
            image = Image.fromarray(image)  # Convert to PIL Image if needed

            # Apply transformation
            augmented_image = transform(image)

            # Save augmented image
            augmented_image_name = f'{counter:04d}.jpg'
            augmented_image_path = os.path.join(aug_dir, augmented_image_name)
            augmented_image.save(augmented_image_path)

            # Store in list for labels.csv
            augmented_rows.append({'image_name': augmented_image_name, 'label': label})

            counter += 1

    # Save labels to labels.csv
    csv_path = os.path.join(out_dir, "augmented_labels.csv")
    df = pd.DataFrame(augmented_rows)
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Path to output augmented images directory', required=True)
   
    # Parse command line args
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    augment(out_dir)
