from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as tv_transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

from models import BasicCNN
from utils.transforms import get_max_hw, basic_transform, \
    get_mean_std_of_channels
from utils.whale_dataset import WhaleDataset
from utils.train_utils import train_loop, val_loop, EarlyStopper
from utils.plots import plot_accuracy, plot_loss


def initial_network():
    """Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""
    ds = WhaleDataset("../data/train", "../data/train.csv")
    classes = ds.int_label_to_cat
    gen = torch.Generator().manual_seed(123)
    dstrain, dsval = random_split(ds, [int(len(ds) * 0.8), int(len(ds) * 0.2)],
                                  generator=gen)
    max_h, max_w = get_max_hw(dstrain)

    dstrain.dataset.transform = basic_transform(max_h, max_w, scale=True)
    dsval.dataset.transform = basic_transform(max_h, max_w, scale=True)

    torch.manual_seed(333)
    trainloader = DataLoader(dstrain, batch_size=16, shuffle=True,
                             num_workers=4)
    valloader = DataLoader(dsval, batch_size=16, shuffle=True, num_workers=4)

    means, devs = get_mean_std_of_channels(dstrain, channels=1)
    dstrain.dataset.transform = tv_transforms.Compose(
        [dstrain.dataset.transform, lambda x: x.float(),
         tv_transforms.Normalize(means, devs)])
    dsval.dataset.transform = tv_transforms.Compose(
        [dsval.dataset.transform, lambda x: x.float(),
         tv_transforms.Normalize(means, devs)])

    trainloader_for_eval = deepcopy(trainloader)

    net = BasicCNN(max_h, max_w).cuda()
    net.train()

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    steps = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    def test(dataloader, acc_only=False):
        correct = 0
        total = 0
        loss_tot = 0.0
        dl = dataloader

        with torch.no_grad():
            for chunk in dl:
                inputs, labels = chunk[0].cuda(), chunk[1].cuda()
                outputs = net(inputs)
                loss = loss_fcn(outputs, labels)
                loss_tot += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = correct / total
        if acc_only:
            return acc
        return loss_tot / len(dl), acc

    total_step = 0

    for epoch in range(100):
        running_train_loss = 0.0
        for step, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            if step % 50 == 49:
                net.eval()
                val_loss, val_acc = test(valloader)
                train_acc = test(trainloader_for_eval, acc_only=True)
                net.train()
                print(
                    f'[{epoch + 1}, {step + 1:5d}] running train loss: {running_train_loss / 50.0:.6f}, train acc {train_acc:.6f}, val loss: {val_loss:.6f}, val acc: {val_acc:.6f}')
                train_losses.append(running_train_loss / 50.0)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                steps.append(total_step + 1)
                running_train_loss = 0.0
                train_acc = 0.0
                val_loss = 0.0
                val_acc = 0.0

            total_step = total_step + 1

    return steps, train_losses, val_losses, train_accs, val_accs


def old_main():
    # cudnn not supported on cs machines (at least phobos)
    torch.backends.cudnn.enabled = False

    steps, train_losses, val_losses, train_accs, val_accs = initial_network()
    os.makedirs("../results", exist_ok=True)

    plt.figure(figsize=(9, 9))
    plt.loglog(steps, train_losses, label="train (running avg)")
    plt.loglog(steps, val_losses, label="val")
    plt.legend()
    plt.title("Whale Model Loss:  Initial Network")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.savefig("../results/initial_network_loss.png", dpi=350)

    plt.figure(figsize=(9, 9))
    plt.loglog(steps, train_accs, label="train")
    plt.loglog(steps, val_accs, label="val")
    plt.legend()
    plt.title("Whale Model Accuracy:  Initial Network")
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.savefig("../results/initial_network_acc.png", dpi=350)


def train(model: nn.Module, dataset: Dataset, params: dict, weights_path: str):
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
    n_classes = params["n_classes"]
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # load dataset
    ds_train, ds_val = random_split(dataset, [int(len(dataset) * 0.8),
                                              int(len(dataset) * 0.2)],
                                    generator=gen)

    # add transforms
    means, devs = get_mean_std_of_channels(ds_train, channels=1)

    ds_train.dataset.transform = basic_transform(img_height, img_width,
                                                 scale=True)
    ds_val.dataset.transform = basic_transform(img_height, img_width,
                                               scale=True)
    ds_train.dataset.transform = tv_transforms.Compose(
        [ds_train.dataset.transform, lambda x: x.float(),
         tv_transforms.Normalize(means, devs)])
    ds_val.dataset.transform = tv_transforms.Compose(
        [ds_val.dataset.transform, lambda x: x.float(),
         tv_transforms.Normalize(means, devs)])

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
                                           train_loader, device, n_classes)
        val_loss, val_acc = val_loop(model, loss_fn, val_loader, device,
                                     n_classes)
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
    torch.backends.cudnn.enabled = False
    ds = WhaleDataset("../data/train", "../data/train.csv")
    n_classes = len(ds.int_label_to_cat)
    model = torchvision.models.efficientnet_b0(
        num_classes=n_classes, n_channels=1)
    # fix to use 1-channel for black/white images
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), bias=False)

    model_name = "effnetb0"
    save_dir = os.path.join("../results", model_name)
    params = {"epochs": 100, "patience": 5, "batch_size": 16,
              "image_height": 256, "image_width": 256, "n_classes": n_classes}

    # setup result dirs
    figures_dir = os.path.join(save_dir, "figures")
    weights_path = os.path.join(save_dir, "model_weights.pth")
    os.makedirs(figures_dir, exist_ok=True)

    results = train(model, ds, params, weights_path)
    loss_pth = os.path.join(figures_dir, "loss.png")
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"],
              loss_pth)
    acc_pth = os.path.join(figures_dir, "accuracy.png")
    plot_accuracy(results["epochs"], results["train_acc"], results["val_acc"],
                  acc_pth)


if __name__ == "__main__":
    main()
