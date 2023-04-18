import torch
from torch import nn


class BasicTwinSiamese(nn.Module):
    """
    Adapted from:
    https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        """
        Initialize siamese network
        :param backbone: nn.Module, backbone network that will
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward_once(self, x):
        output = self.backbone(x)
        output = self.head(output)
        return output

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


if __name__ == "__main__":
    from torchvision.models import efficientnet_b2

    n_features = 512
    backbone = efficientnet_b2(num_classes=n_features)
    head = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, 128))
    siam_net = BasicTwinSiamese(backbone, head)
    img1 = torch.randn((1, 3, 256, 256))
    img2 = torch.randn((1, 3, 256, 256))
    o1, o2 = siam_net(img1, img2)
    print(sum(p.numel() for p in siam_net.parameters() if p.requires_grad))
    print(o1.shape)
    print(torch.max(o1))
    print(torch.min(o1))
