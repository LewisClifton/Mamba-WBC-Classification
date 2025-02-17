from .medmamba import MedMamba

def get_medmamba(num_classes):

    model = MedMamba(num_classes=num_classes)

    return model