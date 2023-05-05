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
from src.utils.transforms import test_alb_transform, train_alb_transform
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
    if "weight_decay" in params:
        optimizer = optim.Adam(model.parameters(),
                               weight_decay=params["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters())

    # load dataset
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


def inceptionResnetV1():
    torch.backends.cudnn.enabled = False
    ds = WhaleDataset("../data/train", "../data/train.csv")
    n_features = 512
    from facenet_pytorch import InceptionResnetV1
    backbone = InceptionResnetV1(pretrained=None, classify=True, num_classes=n_features)

    from src.models.basic_twin_siamese import BasicTwinSiamese
    model = BasicTwinSiamese(backbone, nn.Identity())

    for par in backbone.parameters():
        par.requires_grad = False
    
    for par in backbone.last_linear.parameters():
        par.requires_grad = True
    for par in backbone.last_bn.parameters():
        par.requires_grad = True
    for par in backbone.logits.parameters():
        par.requires_grad = True

    model_name = "inceptionresnetv1_nopretraining_freezeexcept_lastlinear_lastbn_logits_classify160x160_siamese_100epochs_augs"
    save_dir = os.path.join("../results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    params = {"epochs": 100, "patience": 101, "batch_size": 16,
              "image_height": 160, "image_width": 160}

    # setup result dirs
    figures_dir = os.path.join(save_dir, "figures")
    weights_path = os.path.join(save_dir, "model_weights.pth")
    os.makedirs(figures_dir, exist_ok=True)

    results = train_twin_siamese(model, params, weights_path, toRGB=True)
    loss_pth = os.path.join(figures_dir, "loss.png")
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"],
              loss_pth)
    # acc_pth = os.path.join(figures_dir, "acc.png")
    # plot_accuracy(results['epochs'], results['train_acc'], results['val_acc'], acc_pth)
    
    thresh = 0.01
    kaggle_pth = os.path.join(save_dir, "kaggle" + str(thresh) + ".csv")
    from src.evaluation import get_regular_predictions, get_siamese_predictions, get_model_embeddings, create_submission_file
    from src.utils.whale_dataset import TestWhaleDataset
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]
    tf = test_alb_transform(160, 160, means, stds, toRGB=True)
    # testdl = DataLoader(TestWhaleDataset("../data/test", transform=tf), batch_size=16, shuffle=False)
    
    train_ds = WhaleDataset("../data/train", "../data/train.csv", transform=tf)
    test_ds = TestWhaleDataset("../data/test", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    train_outs, test_outs = get_model_embeddings(model, train_loader, test_loader, "cuda:0")
    create_submission_file(get_siamese_predictions(train_outs, test_outs, ds.int_label_to_cat,
                                                   k=5, threshold=thresh), kaggle_pth)


def xception():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    torch.backends.cudnn.enabled = False
    ds = WhaleDataset("../data/train", "../data/train.csv")
    n_features = 1000
    from pretrainedmodels import xception
    backbone = xception(n_features, pretrained="imagenet")

    from src.models.basic_twin_siamese import BasicTwinSiamese
    model = BasicTwinSiamese(backbone, nn.Identity())

    for par in backbone.parameters():
        par.requires_grad = False
    
    for par in backbone.last_linear.parameters():
        par.requires_grad = True
    for par in backbone.bn4.parameters():
        par.requires_grad = True

    model_name = "xception_imagenet_freezeexceptlastlinearbn4_1000feats_299x299_siamese_100epochs_augs"
    save_dir = os.path.join("../results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    params = {"epochs": 100, "patience": 101, "batch_size": 16,
              "image_height": 299, "image_width": 299}

    # setup result dirs
    figures_dir = os.path.join(save_dir, "figures")
    weights_path = os.path.join(save_dir, "model_weights.pth")
    os.makedirs(figures_dir, exist_ok=True)

    results = train_twin_siamese(model, params, weights_path, toRGB=True)
    loss_pth = os.path.join(figures_dir, "loss.png")
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"],
              loss_pth)
    # acc_pth = os.path.join(figures_dir, "acc.png")
    # plot_accuracy(results['epochs'], results['train_acc'], results['val_acc'], acc_pth)
    
    thresh = 0.01
    kaggle_pth = os.path.join(save_dir, "kaggle" + str(thresh) + ".csv")
    from src.evaluation import get_regular_predictions, get_siamese_predictions, get_model_embeddings, create_submission_file
    from src.utils.whale_dataset import TestWhaleDataset
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]
    tf = test_alb_transform(299, 299, means, stds, toRGB=True)
    # testdl = DataLoader(TestWhaleDataset("../data/test", transform=tf), batch_size=16, shuffle=False)
    
    train_ds = WhaleDataset("../data/train", "../data/train.csv", transform=tf)
    test_ds = TestWhaleDataset("../data/test", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    train_outs, test_outs = get_model_embeddings(model, train_loader, test_loader, "cuda:0")
    create_submission_file(get_siamese_predictions(train_outs, test_outs, ds.int_label_to_cat,
                                                   k=5, threshold=thresh), kaggle_pth)


def main():
    from torchvision.models import efficientnet_b2
    torch.backends.cudnn.enabled = False
    n_features = 512
    backbone = efficientnet_b2(num_classes=n_features)
    head = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, 128))
    model = BasicTwinSiamese(backbone, head, dropout_r=0.3)

    model_name = "effnetb2_regularized"
    save_dir = os.path.join("../results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    params = {"epochs": 1000, "patience": 20, "batch_size": 4,
              "image_height": 256, "image_width": 256, "weight_decay": 1e-5}

    # setup result dirs
    figures_dir = os.path.join(save_dir, "figures")
    weights_path = os.path.join(save_dir, "model_weights.pth")
    os.makedirs(figures_dir, exist_ok=True)
    results = train_twin_siamese(model, params, weights_path, toRGB=False)
    loss_pth = os.path.join(figures_dir, "loss.png")
    plot_loss(results["epochs"], results["train_loss"], results["val_loss"],
              loss_pth)


if __name__ == "__main__":
    xception()