from medmnist import BloodMNIST

def get_bloodmnist():
    train_dataset = BloodMNIST(split='train', download=True)
    val_dataset = BloodMNIST(split='val', download=True)

    return train_dataset, val_dataset