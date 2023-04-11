import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Adapted from https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/"""
    
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2, keepdim=True)
        return torch.mean((1.0 - label) * torch.pow(dist, 2.0) + label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2.0))