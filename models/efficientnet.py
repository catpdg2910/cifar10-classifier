import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights_b0 = EfficientNet_B0_Weights.DEFAULT


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = efficientnet_b0(weights=weights_b0)

        self.model.features.requires_grad_(False)

        # Giữ Dropout sẵn có, thay Linear cuối
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def train(self, mode=True):
        super().train(mode)

        if mode:
            for block in self.model.features:
                if not any(p.requires_grad for p in block.parameters()):
                    block.eval()

        return self

    def unfreeze_last_blocks(self, n=3):
        for block in self.model.features[-n:]:
            block.requires_grad_(True)

    def forward(self, x):
        return self.model(x)
