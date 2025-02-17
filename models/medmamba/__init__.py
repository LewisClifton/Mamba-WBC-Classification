from .medmamba import MedMamba
from torchvision import transforms


TRANSFORM_MEDMAMBA = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


def get(num_classes):

    model = MedMamba(num_classes=num_classes)

    return model, TRANSFORM_MEDMAMBA