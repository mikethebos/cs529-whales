import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision.models import efficientnet_b2

from src.models.basic_twin_siamese import BasicTwinSiamese
from src.utils.losses import ContrastiveLoss
from src.utils.plots import plot_loss
from src.utils.train_utils import train_loop, val_loop, twin_siamese_train_loop, \
    twin_siamese_val_loop, EarlyStopper
from src.utils.transforms import basic_alb_transform
from src.utils.whale_dataset import TwinSiameseDataset, WhaleDataset


def train(model: nn.Module, params: dict, weights_path: str):
    """Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # setup dataset
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]

    tf = basic_alb_transform(img_height, img_width, means, stds)
    dataset = WhaleDataset("../data/train", "../data/train.csv",
                           transform=tf)
    ds_train, ds_val = random_split(dataset, [int(len(dataset) * 0.8),
                                              int(len(dataset) * 0.2)],
                                    generator=gen)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

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


def train_twin_siamese(model: nn.Module, params: dict, weights_path: str):
    """Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

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
    optimizer = optim.Adam(model.parameters())

    # load dataset
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]

    tf = basic_alb_transform(img_height, img_width, means, stds)
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
    # train_accs = []
    val_losses = []
    # val_accs = []
    epoch_ls = []
    print_every = 2
    for epoch in range(epochs):
        # train_loss, train_acc = train_loop(model, loss_fn, optimizer,
        #                                    train_loader, device)
        # val_loss, val_acc = val_loop(model, loss_fn, val_loader, device)
        train_loss = twin_siamese_train_loop(model, loss_fn, optimizer,
                                             train_twin_loader, device)
        val_loss = twin_siamese_val_loop(model, loss_fn, val_twin_loader,
                                         device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # train_accs.append(train_acc)
        # val_accs.append(val_acc)
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
    n_features = 128  # backbone is doing feature extraction, not classification
    backbone = efficientnet_b2(num_classes=n_features)
    model = BasicTwinSiamese(backbone)

    model_name = "effnetb2_twinsiamese256x256"
    save_dir = os.path.join("../results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    params = {"epochs": 1000, "patience": 5, "batch_size": 16,
              "image_height": 256, "image_width": 256}

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
