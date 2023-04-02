import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

from utils.whale_dataset import *
from utils.transforms import *
from models import *

def initial_network():
    """Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""
    ds = WhaleDataset("../data/train", "../data/train.csv")
    classes = ds.int_label_to_cat
    gen = torch.Generator().manual_seed(123)
    dstrain, dsval = random_split(ds, [int(len(ds) * 0.8), int(len(ds) * 0.2)], generator=gen)
    max_h, max_w = get_max_hw(dstrain)
    
    dstrain.dataset.transform = basic_transform(max_h, max_w, upscale=True)
    dsval.dataset.transform = basic_transform(max_h, max_w, upscale=True)
    
    torch.manual_seed(333)
    trainloader = DataLoader(dstrain, batch_size=16, shuffle=True, num_workers=4)
    valloader = DataLoader(dsval, batch_size=16, shuffle=True, num_workers=4)

    net = BasicCNN(max_h, max_w).cuda()
    net.train()
    
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    def test(dataloader, acc_only=False):
        correct = 0
        total = 0
        loss_tot = 0.0
        
        with torch.no_grad():
            for chunk in dataloader:
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
        return loss_tot / len(dataloader), acc
    
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
                train_acc = test(trainloader, acc_only=True)
                net.train()
                print(f'[{epoch + 1}, {step + 1:5d}] running train loss: {running_train_loss / 50.0:.6f}, train acc {train_acc:.6f}, val loss: {val_loss:.6f}, val acc: {val_acc:.6f}')
                train_losses.append(running_train_loss / 50.0)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                running_train_loss = 0.0
                train_acc = 0.0
                val_loss = 0.0
                val_acc = 0.0
                
    return train_losses, val_losses, train_accs, val_accs

if __name__ == "__main__":
    # cudnn not supported on cs machines (at least phobos)
    torch.backends.cudnn.enabled = False

    train_losses, val_losses, train_accs, val_accs = initial_network()
    
    import numpy as np
    steps = np.arange(1, len(train_losses) + 1, 1, dtype=np.int64)
    
    import matplotlib.pyplot as plt
    
    import os
    os.makedirs("../results", exist_ok=True)
    
    plt.figure(figsize=(9, 9))
    plt.plot(steps, train_losses, label="train (running avg)")
    plt.plot(steps, val_losses, label="val")
    plt.legend()
    plt.title("Whale Model Loss:  Initial Network")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.savefig("../results/initial_network_loss.png", dpi=350)
    
    plt.figure(figsize=(9, 9))
    plt.plot(steps, train_accs, label="train")
    plt.plot(steps, val_accs, label="val")
    plt.legend()
    plt.title("Whale Model Accuracy:  Initial Network")
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.savefig("../results/initial_network_acc.png", dpi=350)
