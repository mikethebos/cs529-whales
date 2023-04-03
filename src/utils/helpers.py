import torch
import numpy as np
import pandas as pd


def conv2d_output_dim(dim_in: int, kernel_dim: int, stride: int = 1,
                      padding: int = 0, dilation: int = 1) -> int:
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
    return 1 + ((dim_in + (2 * padding) - (
            dilation * (kernel_dim - 1)) - 1) // stride)


def get_top_k(model: torch.nn.Module, img: torch.Tensor,
              int_label_to_cat: pd.Series, k: int = 5):
    """
    Get top k predictions for a given input.
    :param model: torch.nn.Module, classifcation model
    :param img: torch.Tensor, input whale image
    :param int_label_to_cat: pd.Series, contains str label names
    :param k: int, number of top indices to retrieve
    :return top k indices of model outputs
    """
    outputs = model(img).squeeze()  # larger outputs indicate higher probability
    outputs = outputs.detach().numpy()
    top_k_indices = np.argpartition(outputs, -k)[-k:]
    cat_labels = int_label_to_cat[top_k_indices]
    return cat_labels


if __name__ == "__main__":
    from whale_dataset import WhaleDataset
    from torchvision.models import resnet18
    from transforms import *

    ds = WhaleDataset("../../data/train", "../../data/train.csv")
    image, label = ds[0]
    print(label)
    print(ds.get_cat_for_label(label))

    model1 = resnet18(num_classes=len(ds.int_label_to_cat))
    image = image.unsqueeze(0)
    top5 = get_top_k(model1, image, ds.int_label_to_cat)
    print(top5)
