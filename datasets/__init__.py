from torch.utils.data import Dataset

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform, test=False):
        self.dataset = dataset
        self.transform = transform
        self.test=test

    def __getitem__(self, idx):
        if self.test: 
            image, label, image_name = self.dataset[idx]
            image = self.transform(image)
            return image, label, image_name

        image, label = self.dataset[idx]
        image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.dataset)
    

class EnsembleDataset(Dataset):
    def __init__(self, dataset, base_model_transforms):
        """
        Args:
            image_paths (list): List of image file paths.
            transforms_list (list): List of transformations (one per model).
        """
        self.dataset = dataset
        self.base_model_transforms = base_model_transforms  # Different transforms for each model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]

        transformed_images = [transform(image) for transform in self.base_model_transforms]
        return transformed_images
    
    

def get_dataset(dataset_config, dataset_download_dir, test=False):
    """
    Initialise fresh model prior to training

    Args:
        model_type (string): Model type e.g. 'swin'
        num_classes (int): Number of WBC classification classes

    Returns:
        torch.nn.Module: Initialised model ready for training
        dict: Dictionary containing the required data transforms. Use "train"/"val" keys to access training/validation data transforms
    """

    dataset_name = dataset_config['name']

    # Get the required dataset
    if dataset_name == 'chula':
        from .chula import get

        dataset = get(dataset_config, test)

    elif dataset_name == 'bloodmnist':
        from .bloodmnist import get

        # BloodMNIST has predefined train/val/test splits
        if test:
            test_dataset = get(dataset_download_dir, test)
            return test_dataset
        else:
            train_dataset, val_dataset = get(dataset_download_dir, test)
            return train_dataset, val_dataset 

    elif dataset_name == 'foo':
        pass

    return dataset