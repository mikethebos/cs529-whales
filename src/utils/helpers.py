import torch
import numpy as np


def conv2d_output_dim(dim_in: int, kernel_dim: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """
    Get output height or width for a Conv2d layer, 
    where total output shape is (batch_size, out_channels, out_height, out_width)
    :param dim_in: int, incoming height or width
    :param kernel_dim: int, height or width of filter
    :param stride: int, distance to move filter at each step
    :param padding: int, extra entries added to dim_in
    :param dilation: int, distance between start of a filter's cell and the next cell
    :return out, int, the output height or width of the Conv2d layer
    """
    return 1 + ((dim_in + (2 * padding) - (dilation * (kernel_dim - 1)) - 1) // stride)


def get_top_k(model: torch.nn.Module, img: torch.Tensor, k: int = 5):
    """
    Get top k predictions for a given input.
    :param model: torch.nn.Module, classifcation model
    :param img: torch.Tensor, input whale image
    :param k: int, number of top indices to retrieve
    :return top k indices of model outputs
    """
    outputs = model(img)  # larger outputs indicate higher probability
    top_k_indices = np.argpartition(outputs, -k)[-k:]
    return top_k_indices
