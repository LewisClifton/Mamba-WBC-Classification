from dataset import WBC5000dataset
from torch.utils .data import DataLoader
from torchvision import models, transforms

images_path = "data/WBC 5000/"
labels_path = "data/labels.csv"

dataset = WBC5000dataset(images_path, labels_path)
dataloader = DataLoader(dataset, batchsize=16)

pre_trained = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=25),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(p=1.0), 
    transforms.ToTensor()
])