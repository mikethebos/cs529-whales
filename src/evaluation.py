"""
@author Jack Ringer, Mike Adams
Date: 4/14/2023
Description:
File containing code for generating predictions / evaluating models.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import efficientnet_b2
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from models.basic_twin_siamese import BasicTwinSiamese
from utils.transforms import basic_alb_transform
from utils.whale_dataset import TestWhaleDataset, WhaleDataset


def get_siamese_predictions(model: nn.Module, train_loader: DataLoader,
                            test_loader: DataLoader,
                            int_label_to_cat: pd.Series, device: str,
                            k: int = 5, threshold: float = 2.0):
    """
    Use a siamese network to generate predictions of a test dataset.
    :param model: nn.Module, siamese network
    :param train_loader: DataLoader, data model was trained on
    :param test_loader: DataLoader, data to generate predictions for
    :param int_label_to_cat: pd.Series, maps indices to str label names
    :param device: str, either "cpu" or "cuda:0"
    :param k: int, generate top-k predictions for each input
    :param threshold: float, cutoff at which to predict "new_whale"
        (lower value => more likely to predict new_whale)
    :return: 2D array predictions, where predictions[i] gives a list of k
        strings indicating the top k predictions for test image i
    """
    predictions = [[""] * k for _ in range(len(test_loader))]
    for i, test_image in tqdm(enumerate(test_loader)):
        test_image = test_image.to(device)
        distances = {}
        for j, (train_image, label_tensor) in enumerate(train_loader):
            label_idx = label_tensor.item()
            train_image = train_image.to(device)
            out_1, out_2 = model(test_image, train_image)
            distance = F.pairwise_distance(out_1, out_2, keepdim=False).item()
            distances[label_idx] = distances.get(label_idx, []) + [distance]
        # minimum average distance corresponds to top score
        avg_distances = {key: sum(values) / len(values) for key, values in
                         distances.items()}
        top_k_indices = sorted(avg_distances, key=avg_distances.get)[:k]
        k_cat_labels = int_label_to_cat[top_k_indices]

        # if score falls below a certain threshold predict "new_whale"
        new_whale_inserted = False
        p = 0
        for w in range(k):
            if new_whale_inserted or \
                    avg_distances[top_k_indices[p]] < threshold:
                predictions[i][w] = k_cat_labels[p]
                p += 1
            else:
                predictions[i][w] = "new_whale"
                new_whale_inserted = True
    return predictions


def main(device: str):
    torch.backends.cudnn.enabled = False

    # load dataset/dataloader
    image_height = 256
    image_width = 256
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]
    tf = basic_alb_transform(image_height, image_width, means, stds)
    train_ds = WhaleDataset("../data/train", "../data/train.csv", transform=tf)
    test_ds = TestWhaleDataset("../data/test", transform=tf)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    # load model and trained weights
    weights_path = "../results/effnetb2_twinsiamese256x256/model_weights.pth"
    n_features = 128
    backbone = efficientnet_b2(num_classes=n_features)
    model = BasicTwinSiamese(backbone)
    model.to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device(device)))

    # get predictions
    predictions = get_siamese_predictions(model, train_loader, test_loader,
                                          train_ds.int_label_to_cat, device)
    print(predictions)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch_device = "cuda:0"
    else:
        torch_device = "cpu"
    main(torch_device)
