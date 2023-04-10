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
    :param dataloader: Torch DataLoader, contains test data
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


def twin_siamese_val_loop(model: nn.Module, loss_fn, train_dataloader: DataLoader, test_dataloader: DataLoader, device: str):
    """
    Carry out an iteration (single epoch) of validation of twin siamese NN. Will not update
    model weights
    :param model: PyTorch model
    :param loss_fn: Loss function
    :param train_dataloader: Torch DataLoader, contains training data in normal form
    :param test_dataloader: Torch DataLoader, contains test data in normal form
    :param device: str, torch device to put tensors on
    :return: float, the total loss over the epoch
    """
    model.eval()
    loss_epoch = 0
    # correct = 0
    with torch.no_grad():
        for i, (model_test_in, test_labels) in enumerate(test_dataloader):
            for j, (model_train_in, train_labels) in enumerate(train_dataloader):
                # get inputs/targets
                model_test_in, test_labels, model_train_in, train_labels = model_test_in.to(device), test_labels.to(device), model_train_in.to(device), train_labels.to(device)
                labels = torch.eq(test_labels, train_labels).long().to(device)
                output_one, output_two = model(model_test_in, model_train_in)

                loss = loss_fn(output_one, output_two, labels)
                loss_epoch += loss.item()
                # _, predictions = torch.max(outputs, 1)
                # correct += torch.sum(torch.eq(predictions, labels)).item()
        loss_epoch /= len(train_dataloader) * len(test_dataloader)
        # acc = correct / len(dataloader.dataset)
    # return loss_epoch, acc
    return loss_epoch


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
