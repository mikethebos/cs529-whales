import torch
from torch import nn

from src.utils.helpers import conv2d_output_dim

class BasicTwinSiamese(nn.Module):
    """Adapted from https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/"""
    
    def __init__(self, input_height, input_width):
        super().__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        conv1_h, conv1_w = conv2d_output_dim(input_height, 11, stride=4), conv2d_output_dim(input_width, 11, stride=4)
        pool1_h, pool1_w = conv2d_output_dim(conv1_h, 3, stride=2), conv2d_output_dim(conv1_w, 3, stride=2)
        
        conv2_h, conv2_w = conv2d_output_dim(pool1_h, 5, stride=1), conv2d_output_dim(pool1_w, 5, stride=1)
        pool2_h, pool2_w = conv2d_output_dim(conv2_h, 2, stride=2), conv2d_output_dim(conv2_w, 2, stride=2)
        
        conv3_h, conv3_w = conv2d_output_dim(pool2_h, 3, stride=1), conv2d_output_dim(pool2_w, 3, stride=1)
        
        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384 * conv3_h * conv3_w, 768),
            nn.ReLU(inplace=True),
            
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,128)
        )
        
    def forward_once(self, x):
        output = self.cnn1(x)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        return output
    
    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
    
if __name__ == "__main__":
    m = BasicTwinSiamese(400, 400).cuda()  # ensure no OOM errors