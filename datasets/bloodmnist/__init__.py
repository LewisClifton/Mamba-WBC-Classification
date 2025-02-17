import os

from medmnist import BloodMNIST

def get(dataset_download_dir, test=False):

    # Set up path for the dataset 
    if dataset_download_dir:
        if not os.path.isdir(dataset_download_dir):
            os.mkdir(dataset_download_dir)

        # Get the dataset
        if not test:
            train_dataset = BloodMNIST(split='train', download=True, size=224, root=dataset_download_dir)
            val_dataset = BloodMNIST(split='val', download=True, size=224, root=dataset_download_dir)
            return train_dataset, val_dataset
        else: 
            test_dataset = BloodMNIST(split='test', download=True, size=224, root=dataset_download_dir)
            return test_dataset
        
    else:

        # Get the dataset
        if not test:
            train_dataset = BloodMNIST(split='train', download=True, size=224)
            val_dataset = BloodMNIST(split='val', download=True, size=224)
            return train_dataset, val_dataset
        else: 
            test_dataset = BloodMNIST(split='test', download=True, size=224)
            return test_dataset