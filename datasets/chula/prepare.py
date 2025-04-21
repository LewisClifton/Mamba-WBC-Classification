import pandas as pd
import os
from PIL import Image
import os
import argparse
import numpy as np

import torchvision.transforms as transforms

# (Execute this script from project root)

def center_crop(image, crop_size=(175, 175)):
    """
    Center crop an image for a given size.

    Args:
        image (Image): Image
        crop_size (tuple): Crop size

    Returns:
        Image: Center cropped image
    """

    # Crop image at center (https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil)
    width, height = image.size
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return image.crop((left, top, right, bottom))


def center_crop_all(images_dir):
    """
    Center crop all images in the dataset, saving to existing locations.

    Args:
        images_dir (string): Path to dataset image directory
    """

    print('Cropping images...')
    files = os.listdir(images_dir)

    # Crop all images
    for file in files:
        if file == 'augmented': continue # ignore 
        file = os.path.join(images_dir, file)

        # Crop and save image
        image = Image.open(file)
        cropped_image = center_crop(image)
        cropped_image.save(file)


def clean_labels(labels_path):
    """
    Clean the dataset labels, removing unwanted column, correcting class name typos, removing unwanted classes.

    Args:
        labels_path (string): Path of dataset labels.csv
    """

    print('Cleaning labels...')

    # Read labels
    labels = pd.read_csv(labels_path)

    # Remove and rename columns
    labels = labels[['Order.jpg', 'Summary by 5 experts']].rename(columns={'Order.jpg': 'name', 'Summary by 5 experts': 'label'})
    
    # Fix typos in labels
    corrected_labels = {
        'Eosinophill': 'Eosinophil',
        'Monocyte ': 'Monocyte',
        'SNE\t': 'SNE',
        'Myeolblast': 'Myeloblast',
        'Atypical lymphocyte': 'Atypical Lymphocyte', 
        'Smudge cell': 'Smudge Cell',
        'Giant platelet': 'Giant Platelet'
    }
    labels['label'] = labels['label'].replace(corrected_labels).str.strip()

    # Only keep required wbc types
    wbc_types=['SNE', 'Lymphocyte', 'Monocyte', 'BNE', 'Eosinophil', 'Myeloblast', 'Basophil', 'Metamyelocyte']
    labels = labels[labels['label'].isin(wbc_types)]

    return labels


def split_dataset(dataset, test_frac=0.2, val_frac=None): # val_frac=0.1
    """
    Split the dataset into train, validaiton (if required) and test split.

    Args:
        dataset (pandas.Dataframe): Dataset labels
        test_frac (float): Fraction of dataset used for test set
        test_frac (float): Fraction of train set used for validation set

    Return:
        pandas.DataFrame: Train split
        pandas.DataFrame or None: Validation split (if required)
        pandas.DataFrame: Test split
    """

    # First, sample the test set
    test = dataset.sample(frac=test_frac, random_state=41)
    remain = dataset.drop(test.index)

    if val_frac:
        # Compute validation fraction relative to the remaining data
        val_relative_frac = val_frac / (1 - test_frac)
        val = remain.sample(frac=val_relative_frac, random_state=41)
        train = remain.drop(val.index)

        return train, val, test

    else:

        return train, None, test


def augment(images_dir, chula_train):
    """
    Augment the dataset and return the updated labels.csv now containing the labels for generated images.

    Args:
        images_dir (string): Path to dataset images directory
        chula_train (pandas.Dataframe): The original dataset labels prior to augmentation

    Return:
        pandas.DataFrame: Updated labels set including the labels for augmented data
    """

    print('Starting augmentation...')

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

    # Create augmented directory
    augmented_dir = os.path.join(images_dir, 'augmented/')
    os.makedirs(augmented_dir, exist_ok=True)

    # Ensure existing augmented images are cleared
    for file in os.listdir(augmented_dir):
        os.remove(os.path.join(augmented_dir, file))

    counter = 5001
    augmented_rows = []

    # median = int(np.median(chula_train['label'].value_counts().values))

    # Get class frequencies
    value_counts = chula_train['label'].value_counts()

    # Iterate through the classes to augment each if necessary
    for label, count in value_counts.items():
        diff = 1000 - count # this is found to be the best augmentation target

        # diff = int(count * 1.50)

        # If already sufficient samples in this class, skip
        if diff <= 0:
            print(f'Label: {label}, Original {count} Augmentations: {0}, Total: {count+diff}')
            continue
        print(f'Label: {label}, Augmentations: {diff}, Total: {count+diff}')

        # Get all images of this label
        label_images = chula_train[chula_train['label'] == label]['name'].tolist()
        num_images = len(label_images)

        # Produce as many samples as needed to reach the augmentation target
        for i in range(diff):
            
            # Get the source image for augmentation
            image_name = label_images[i % num_images] # fairly augment each original image 
            image_path = os.path.join(images_dir, image_name)

            # Open and center crop the original image
            image = Image.open(image_path)

            # Apply augmentation
            augmented_image = transform(image)

            # Save augmented image
            augmented_image_name = f'{str(counter).zfill(6)}.jpg'
            augmented_image.save(os.path.join(augmented_dir, augmented_image_name))

            # Add name and label to dataframe
            augmented_rows.append({'name': f'augmented/{augmented_image_name}', 'label': label})

            counter += 1

    # Create final labels dataframe
    augmented_df = pd.DataFrame(augmented_rows)
    chula_augmented_train = pd.concat([chula_train, augmented_df], ignore_index=True)

    # Shuffle dataset
    chula_augmented_train = chula_augmented_train.sample(frac=1, random_state=42).reset_index(drop=True)

    return chula_augmented_train


if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, help='Path to Chula images directory', required=True)
    parser.add_argument('--labels_dir', type=str, help='Path to directory containing labels.csv', required=True)

   
    # Parse command line args
    args = parser.parse_args()
    images_dir = args.images_dir
    labels_dir = args.labels_dir

    # # Clean labels and save
    # labels_path = os.path.join(labels_dir, 'labels.csv')
    # labels = clean_labels(labels_path)
    # labels.to_csv(os.path.join(labels_dir, 'labels_clean.csv'), index=False)

    # # # Center crop images
    # center_crop_all(images_dir)

    # # # Split labels into train and test set save
    # chula_train, _, chula_test = split_dataset(labels)
    # chula_train.to_csv(os.path.join(labels_dir, 'labels_train.csv'), index=False)
    # chula_test.to_csv(os.path.join(labels_dir, 'labels_test.csv'), index=False)

    # Apply augmentations to training data to reduce imbalance and save their labels

    chula_train = pd.read_csv(os.path.join(labels_dir, 'labels_train.csv')) 
    
    chula_augmented_train = augment(images_dir, chula_train)
    chula_augmented_train.to_csv(os.path.join(labels_dir, 'labels_train_augmented.csv'), index=False)
