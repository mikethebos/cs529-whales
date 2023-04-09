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
    :return: tuple[float], the total running loss, running acc over the epoch
    """
    model.train()
    loss_epoch = 0
    correct = 0
    for i, (model_in, labels) in enumerate(dataloader):
        # get inputs/targets
        model_in, labels = model_in.to(device), labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(model_in)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        _, predictions = torch.max(outputs, 1)
        correct += torch.sum(torch.eq(predictions, labels)).item()
    loss_epoch /= len(dataloader)
    acc = correct / len(dataloader.dataset)
    return loss_epoch, acc


def twin_siamese_train_loop(model: nn.Module, loss_fn, optimizer: Optimizer,
               dataloader: DataLoader,
               device: str):
    """
    Carry out an iteration (single epoch) of training of twin siamese NN. Will update model weights
    :param model: PyTorch model
    :param loss_fn: Loss function
    :param optimizer: Optimizer to use
    :param dataloader: Torch DataLoader, contains training data in twin siamese form
    :param device: str, torch device to put tensors on
    :return: float, the total running loss over the epoch
    """
    model.train()
    loss_epoch = 0
    # correct = 0
    for i, (model_in_one, model_in_two, labels) in enumerate(dataloader):
        # get inputs/targets
        model_in_one, model_in_two, labels = model_in_one.to(device), model_in_two.to(device), labels.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output_one, output_two = model(model_in_one, model_in_two)
        loss = loss_fn(output_one, output_two, labels)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # _, predictions = torch.max(outputs, 1)
        # correct += torch.sum(torch.eq(predictions, labels)).item()
    loss_epoch /= len(dataloader)
    # acc = correct / len(dataloader.dataset)
    # return loss_epoch, acc
    return loss_epoch


def val_loop(model: nn.Module, loss_fn, dataloader: DataLoader, device: str):
    """
    Carry out an iteration (single epoch) of validation. Will not update
    model weights
    :param model: PyTorch model
    :param loss_fn: Loss function
    :param dataloader: Torch DataLoader, contains training data
    :param device: str, torch device to put tensors on
    :return: tuple[float], the total loss, acc over the epoch
    """
    model.eval()
    loss_epoch = 0
    correct = 0
    with torch.no_grad():
        for i, (model_in, labels) in enumerate(dataloader):
            # get inputs/targets
            model_in, labels = model_in.to(device), labels.to(device)
            outputs = model(model_in)

            loss = loss_fn(outputs, labels)
            loss_epoch += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct += torch.sum(torch.eq(predictions, labels)).item()
        loss_epoch /= len(dataloader)
        acc = correct / len(dataloader.dataset)
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
