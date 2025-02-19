import pandas as pd
import os
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

# Execute script from project root


def clean_labels(labels_path):
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
        # labels = labels[labels['label'].isin(wbc_types)]

        # Remove rows where the label is "LQI"
        labels = labels[labels['label'] != 'LQI']

        return labels


def train_test_split(dataset, frac=0.2):
    test = dataset.sample(frac=frac, random_state=41)  # 20% of the data
    train = dataset.drop(test.index)

    return train, test


def augment(images_dir):
    
    # Augmentation transform
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=360),        
        transforms.RandomAffine(
            degrees=0,                              
            translate=(0.1, 0.1),                    
            scale=(0.9, 1.1),                   
            shear=5                               
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),                
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02), 
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])


    # Create augmented directory
    augmented_dir = os.path.join(images_dir, 'augmented/')
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
    else:
        for file in os.listdir(augmented_dir):
            os.remove(os.path.join(augmented_dir, file))
        
    counter = 5001
    augmented_rows = []

    for label, count in chula_train['label'].value_counts().items():
        diff = 500 - count
        if diff < 0: continue

        for i in range(diff):
            # Get random image of this label
            random_image = chula_train[chula_train['label'] == label].sample(n=1, random_state=None).iloc[0]['name']
            image = Image.open(os.path.join(images_dir, random_image))

            # Apply transformation
            augmented_image = transform(image)

            # Save to the augmented subdirectory
            augmented_image_name = f'{(6-len(str(counter)))*"0"}{counter}.jpg'
            augmented_image.save(os.path.join(augmented_dir, augmented_image_name))
            
            # Add name and label to dataframe
            augmented_rows.append({'name' : f'augmented/{augmented_image_name}', 'label' : label})

            counter += 1

    # Create final labels dataframe
    augmented_df = pd.DataFrame(augmented_rows)
    chula_augmented_train = pd.concat([chula_train, augmented_df], ignore_index=True)

    print(chula_augmented_train['label'].value_counts())

    # Save names and labels dataframe
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

    # Clean labels and save
    labels_path = os.path.join(labels_dir, 'labels.csv')
    labels = clean_labels(labels_path)
    labels.to_csv(os.path.join(labels_dir, 'labels_clean.csv'), index=False)

    # Split labels into train and test set save
    chula_train, chula_test = train_test_split(labels)
    chula_train.to_csv(os.path.join(labels_dir, 'labels_train.csv'), index=False)
    chula_test.to_csv(os.path.join(labels_dir, 'labels_test.csv'), index=False)

    # Apply augmentations to training data to reduce imbalance and save their labels
    chula_augmented_train = augment(images_dir)
    chula_augmented_train.to_csv(os.path.join(labels_dir, 'labels_train_augmented.csv'), index=False)
