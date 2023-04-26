import numpy as np
import pandas as pd
import torch


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


def get_top_k(model: torch.nn.Module, imgs: torch.Tensor,
              int_label_to_cat: pd.Series, k: int = 5) -> "list[list]":
    """
    Get top k predictions for a given input.
    :param model: torch.nn.Module, classifcation model
    :param imgs: torch.Tensor, input whale images
    :param int_label_to_cat: pd.Series, contains str label names
    :param k: int, number of top indices to retrieve
    :return top k indices of model outputs
    """
    outputs = model(imgs)  # larger outputs indicate higher probability
    outputs = outputs.detach().cpu().numpy()
    top_k_indices = (np.argpartition(outputs, -k, axis=1)[:, -k:]).tolist()
    cat_labels = list(map(lambda l: list(map(lambda i: int_label_to_cat[i], l)),
                          top_k_indices))
    return cat_labels


def get_model_params(model: torch.nn.Module):
    """
    Get the number of trainable parameters for a model
    :param model: nn.Module, PyTorch model
    :return: int, number of trainable params in model
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def freeze_effnet_weights(effnet_model: torch.nn.Module, in_features: int,
                          out_features: int):
    """
    Freeze the weights of an efficient-net model
    :param effnet_model: nn.Module, torch efficientnet model
    :param in_features: int, input features to classification head
    :param out_features: int, output features from classification head
    :return: model, effnet with all layers frozen except head
    """
    for name, param in effnet_model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # Change the final classification head.
    effnet_model.classifier[1] = torch.nn.Linear(in_features=in_features,
                                                 out_features=out_features)
    return effnet_model


if __name__ == "__main__":
    from torchvision.models import efficientnet_b2

    model1 = efficientnet_b2(num_classes=512)
    print(model1)
    print("trainable params before freeze:", get_model_params(model1))
    model1 = freeze_effnet_weights(model1, 512, 256)
    print("after:", get_model_params(model1))
    for name, param in model1.named_parameters():
        if param.requires_grad:
            print(name)
