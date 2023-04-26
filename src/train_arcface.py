"""
@author Jack Ringer, Mike Adams
Date: 4/21/2023
Description:
Contains code used to train classification models
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from pytorch_metric_learning import losses

from src.utils.plots import plot_loss, plot_accuracy
from src.utils.train_utils import train_loop, val_loop, EarlyStopper
from src.utils.transforms import test_alb_transform, train_alb_transform
from src.utils.whale_dataset import WhaleDataset
from src.models.feature_extractor import FeatureExtractor


def train(model: nn.Module, params: dict, weights_path: str,
          toRGB: bool = False):
    """
    Train model
    :param model: nn.Module, feature extractor to train
    :param params: dict, hyper-params to use
    :param weights_path: str, path to save best model weights to
    :param toRGB: bool, specify whether to convert images to RGB
    :return: training and validation losses
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print("Using device:", device)
    model.to(device)

    # set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    gen = torch.Generator().manual_seed(42)

    # setup params
    img_height, img_width = params["image_height"], params["image_width"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    patience = params["patience"]
    early_stopper = EarlyStopper(patience=patience, min_delta=0)

    optimizer = optim.Adam(model.parameters())

    # setup dataset
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]

    train_tf = train_alb_transform(img_height, img_width, means, stds,
                                   toRGB=toRGB)
    test_tf = test_alb_transform(img_height, img_width, means, stds,
                                 toRGB=toRGB)
    dataset = WhaleDataset("../data/train", "../data/train.csv",
                           transform=None)
    ds_train, ds_val = random_split(dataset, [int(len(dataset) * 0.8),
                                              int(len(dataset) * 0.2)],
                                    generator=gen)
    ds_train.dataset.transform = train_tf
    ds_val.dataset.transform = test_tf
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    # setup loss function
    num_classes = len(dataset.int_label_to_cat)
    embed_size = params["embed_size"]
    loss_fn = losses.ArcFaceLoss(num_classes, embed_size)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epoch_ls = []
    print_every = 2
    for epoch in range(epochs):
        train_loss, train_acc = train_loop(model, loss_fn, optimizer,
                                           train_loader, device)
        val_loss, val_acc = val_loop(model, loss_fn, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        epoch_ls.append(epoch)
        if epoch % print_every == 0:
            print(
                f'{epoch + 1} train loss: {train_loss / 50.0:.6f}, train acc {train_acc:.6f}, val loss: {val_loss:.6f}, val acc: {val_acc:.6f}')
        if val_loss < early_stopper.min_validation_loss:
            # save model with the lowest validation accuracy
            torch.save(model.state_dict(), weights_path)
        if early_stopper.early_stop(val_loss):
            break
    results = {"epochs": epoch_ls, "train_loss": train_losses,
               "val_loss": val_losses, "train_acc": train_accs,
               "val_acc": val_accs}
    return results


def main():
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
    torch.backends.cudnn.enabled = False
    n_features = 512
    embed_size = 256
    backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    backbone.classifier[1] = nn.Linear(in_features=1408,
                                       out_features=n_features)
    head = nn.Sequential(nn.ReLU(), nn.Linear(n_features, embed_size))
    model = FeatureExtractor(backbone, head, 0.3)

    model_name = "effnetb2_arcface"
    save_dir = os.path.join("../results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    params = {"epochs": 1000, "patience": 20, "batch_size": 16,
              "image_height": 256, "image_width": 256, "embed_size": embed_size}

    # setup result dirs
    figures_dir = os.path.join(save_dir, "figures")
    weights_path = os.path.join(save_dir, "model_weights.pth")
    os.makedirs(figures_dir, exist_ok=True)

    results = train(model, params, weights_path, toRGB=True)
    loss_pth = os.path.join(figures_dir, "loss.png")
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"],
              loss_pth)
    acc_pth = os.path.join(figures_dir, "acc.png")
    plot_accuracy(results['epochs'], results['train_acc'], results['val_acc'],
                  acc_pth)


if __name__ == "__main__":
    main()
