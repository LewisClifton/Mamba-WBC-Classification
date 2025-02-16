from transformers import AutoModelForImageClassification
        
# Need a wrapper to remove the dict wrapping of mambavision.forward
class MambaVisionWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)

    def forward(self, x):
        return self.model(x)['logits']


def get_mambavision(num_classes):
    model = MambaVisionWrapper()

    # To do: change heda

    return model