"""
@author Jack Ringer, Mike Adams
Date: 4/6/2023
Description: This file contains various functions used to organize the
training of models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train_loop(model: nn.Module, loss_fn, optimizer: Optimizer,
               dataloader: DataLoader,
               device: str):
    """
    Carry out an iteration (single epoch) of training. Will update model weights
    :param model: PyTorch model
    :param loss_fn: Loss function
    :param optimizer: Optimizer to use
    :param dataloader: Torch DataLoader, contains training data
    :param device: str, torch device to put tensors on
    :return: float, the total loss over the epoch
    """
    model.train()
    loss_epoch = 0
    correct = 0
    total = 0
    for i, batch in enumerate(dataloader):
        # get inputs/targets
        model_in, labels = batch[0].to(device), batch[1].to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predictions = model.forward(model_in)
        loss = loss_fn(labels, predictions)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        total += len(batch)
        correct += torch.sum(torch.eq(predictions, labels)).item()
    loss_epoch /= len(dataloader)
    acc = correct / total
    return loss_epoch, acc


def val_loop(model: nn.Module, loss_fn, dataloader: DataLoader, device: str):
    """
    Carry out an iteration (single epoch) of validation. Will not update
    model weights
    :param model: PyTorch model
    :param loss_fn: Loss function
    :param dataloader: Torch DataLoader, contains training data
    :param device: str, torch device to put tensors on
    :return: float, the total loss over the epoch
    """
    model.eval()
    loss_epoch = 0
    correct = 0
    total = 0
    with torch.no_grad:
        for i, batch in enumerate(dataloader):
            # get inputs/targets
            model_in, labels = batch[0].to(device), batch[1].to(device)
            predictions = model.forward(model_in)
            loss = loss_fn(labels, predictions)
            loss_epoch += loss.item()
            total += len(batch)
            correct += torch.sum(torch.eq(predictions, labels)).item()
        loss_epoch /= len(dataloader)
        acc = correct / total
    return loss_epoch, acc


# taken from:
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """
    Class to use for early stopping.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
