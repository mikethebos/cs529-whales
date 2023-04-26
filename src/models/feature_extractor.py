"""
@author Jack Ringer, Mike Adams
Date: 4/23/2023
Description:
Class used for a pure feature extractor. Intended for use with ArcFace loss
or something similar.
"""
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module,
                 dropout_r: float = 0.0):
        """
        Initialize feature extractor network
        :param backbone: nn.Module, backbone network that will
        :param head: nn.Module, head portion of model to make prediction
        :param dropout_r: float (optional), dropout rate to use between backbone
            and head
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        o = self.backbone(x)
        o = self.dropout(o)
        o = self.head(o)
        return o


if __name__ == "__main__":
    import torch
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

    n_features = 512
    embed_size = 256
    backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    backbone.classifier[1] = nn.Linear(in_features=1408,
                                       out_features=n_features)
    head = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, 128))
    model = FeatureExtractor(backbone, head)
    img1 = torch.randn((1, 3, 256, 256))
    y1 = model(img1)
    y11 = model(img1)
    print(torch.min(y1))
    print(torch.min(y11))
    model.eval()
    y2 = model(img1)
    print(torch.min(y2))
