"""
@author: Jack Ringer, Mike Adams
Date: 4/7/2023
Description: File containing various plotting-related functions
"""

import matplotlib.pyplot as plt


def plot_loss(epochs, train_losses, val_losses, save_pth: str):
    plt.figure(figsize=(9, 9))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.yscale('log')
    plt.legend()
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(save_pth, dpi=350)


def plot_accuracy(epochs, train_accs, val_accs, save_pth: str):
    plt.figure(figsize=(9, 9))
    plt.plot(epochs, train_accs, label="train")
    plt.plot(epochs, val_accs, label="val")
    plt.yscale('log')
    plt.legend()
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(save_pth, dpi=350)
