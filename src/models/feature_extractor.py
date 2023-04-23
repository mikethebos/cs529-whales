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
                 dropout_r: float = -1.0):
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
        self.dropout = None
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        o = self.backbone(x)
        if self.dropout is not None:
            o = self.dropout(o)
        o = self.head(o)
        return o
