"""
@author Jack Ringer, Mike Adams
Date: 4/21/2023
Description:
Contains code used to siamese (similarity-based) models
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from src.models.basic_twin_siamese import BasicTwinSiamese
from src.utils.losses import ContrastiveLoss
from src.utils.plots import plot_loss
from src.utils.train_utils import twin_siamese_train_loop, \
    twin_siamese_val_loop, EarlyStopper
from src.utils.transforms import test_alb_transform
from src.utils.whale_dataset import TwinSiameseDataset, WhaleDataset


def train_twin_siamese(model: BasicTwinSiamese, params: dict, weights_path: str,
                       toRGB: bool = False):
    """
    Train a twin siamese network. Adapted from:
     https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    :param model: BasicTwinSiamese, twin siamese network
    :param params: dict, hyper-params to use for training
    :param weights_path: str, path to save best model weights to
    :param toRGB: bool, whether to convert images to RGB
    :return: dict containing epoch, training loss, validation loss
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
    loss_fn = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(),
                           weight_decay=params["weight_decay"])

    # load dataset
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]

    tf = test_alb_transform(img_height, img_width, means, stds,
                            toRGB=toRGB)
    dataset = WhaleDataset("../data/train", "../data/train.csv",
                           transform=tf)
    ds_train, ds_val = random_split(dataset, [int(len(dataset) * 0.8),
                                              int(len(dataset) * 0.2)],
                                    generator=gen)

    # train twin dataset
    ds_train_twin = TwinSiameseDataset(ds_train)
    train_twin_loader = DataLoader(ds_train_twin, batch_size=batch_size,
                                   shuffle=True)

    ds_val_twin = TwinSiameseDataset(ds_val)
    val_twin_loader = DataLoader(ds_val_twin, batch_size=batch_size,
                                 shuffle=True)

    train_losses = []
    val_losses = []
    epoch_ls = []
    print_every = 2
    for epoch in range(epochs):
        train_loss = twin_siamese_train_loop(model, loss_fn, optimizer,
                                             train_twin_loader, device)
        val_loss = twin_siamese_val_loop(model, loss_fn, val_twin_loader,
                                         device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_ls.append(epoch)
        if epoch % print_every == 0:
            print(
                f'{epoch + 1} train loss: {train_loss / 50.0:.6f}, val loss: {val_loss:.6f}')
        if val_loss < early_stopper.min_validation_loss:
            # save model with the lowest validation accuracy
            torch.save(model.state_dict(), weights_path)
        if early_stopper.early_stop(val_loss):
            break
    results = {"epochs": epoch_ls, "train_loss": train_losses,
               "val_loss": val_losses}
    return results


def main():
    torch.backends.cudnn.enabled = False
    n_features = 512
    from facenet_pytorch import InceptionResnetV1
    backbone = InceptionResnetV1(pretrained="vggface2", classify=False,
                                 num_classes=n_features)
    head = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, 128))
    model = BasicTwinSiamese(backbone, head)

    model_name = "inceptionres_finetune"
    save_dir = os.path.join("../results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    params = {"epochs": 1000, "patience": 20, "batch_size": 16,
              "image_height": 256, "image_width": 256, "weight_decay": 0}

    # setup result dirs
    figures_dir = os.path.join(save_dir, "figures")
    weights_path = os.path.join(save_dir, "model_weights.pth")
    os.makedirs(figures_dir, exist_ok=True)
    results = train_twin_siamese(model, params, weights_path)
    loss_pth = os.path.join(figures_dir, "loss.png")
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"],
              loss_pth)


if __name__ == "__main__":
    main()
