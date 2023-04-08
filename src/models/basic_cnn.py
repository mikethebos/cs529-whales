import torch
from torch import nn
from torch.nn.functional import relu

from utils.helpers import conv2d_output_dim

class BasicCNN(nn.Module):
    """Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""
    def __init__(self, input_height, input_width):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # assume grayscale
        out1_h, out1_w = conv2d_output_dim(input_height, 5), conv2d_output_dim(input_width, 5)
        self.pool = nn.MaxPool2d(2, 2)
        outpool_h, outpool_w = conv2d_output_dim(out1_h, 2, stride=2), conv2d_output_dim(out1_w, 2, stride=2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        out2_h, out2_w = conv2d_output_dim(outpool_h, 5), conv2d_output_dim(outpool_w, 5)
        pool2_h, pool2_w = conv2d_output_dim(out2_h, 2, stride=2), conv2d_output_dim(out2_w, 2, stride=2)
        self.fc1 = nn.Linear(8 * pool2_h * pool2_w, 550)
        self.fc2 = nn.Linear(550, 4251)  # num classes = 4251
        
    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # keep batch dim
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x
