"""
@author Jack Ringer, Mike Adams
Date: 4/14/2023
Description:
File containing code for generating predictions / evaluating models.
"""

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b2
from tqdm import tqdm

from models.basic_twin_siamese import BasicTwinSiamese
from utils.helpers import get_top_k
from utils.transforms import basic_alb_transform
from utils.whale_dataset import TestWhaleDataset, WhaleDataset


def get_regular_predictions(model: nn.Module, test_loader: DataLoader, int_label_to_cat: pd.Series, device: str,
                            k: int = 5):
    model.eval()
    out = {}
    with torch.no_grad():
        # TestWhaleDataset only
        for i, (test_imgs, test_fnames) in tqdm(enumerate(test_loader)):
            test_imgs = test_imgs.to(device)
            top_k = get_top_k(model, test_imgs, int_label_to_cat, k=k)
            out.update({test_fnames[j]: li for j, li in enumerate(top_k)})
    return out


def get_siamese_backbone_outs(model: BasicTwinSiamese, train_loader: DataLoader, test_loader: DataLoader, device: str):
    model.eval()
    train_outs = [(0, 0)] * len(train_loader)
    print("Getting train outputs...")
    with torch.no_grad():
        for i, (train_image, label_idx) in tqdm(enumerate(train_loader)):
            train_image = train_image.to(device)
            label_idx = label_idx.item()
            out = model.forward_once(train_image)
            train_outs[i] = (out, label_idx)

    test_outs = {}
    print("Getting test outputs...")
    with torch.no_grad():
        # TestWhaleDataset only
        for i, (test_image, image_filename) in tqdm(enumerate(test_loader)):
            image_filename = image_filename[0]  # gets wrapped by loader
            test_image = test_image.to(device)
            out = model.forward_once(test_image)
            test_outs[image_filename] = out

    return train_outs, test_outs


def get_siamese_predictions(train_outs: "list[tuple]",
                            test_outs: dict,
                            int_label_to_cat: pd.Series, device: str,
                            k: int = 5, threshold: float = 2.0):
    """
    Use a siamese network to generate predictions of a test dataset.
    :param train_outs: list[tuple], train_outs[i] = extracted features, label
    :param test_outs: dict, maps filenames to extracted features
    :param int_label_to_cat: pd.Series, maps indices to str label names
    :param device: str, either "cpu" or "cuda:0"
    :param k: int, generate top-k predictions for each input
    :param threshold: float, cutoff at which to predict "new_whale"
        (lower value => more likely to predict new_whale)
    :return: dictionary predictions where predictions[fname] gives the list of k predictions for
        a given image file from the test data
    """
    predictions = {}
    for img_fname in tqdm(test_outs.keys()):
        distances = {}
        test_out = test_outs[img_fname]
        for train_out, label_idx in train_outs:
            distance = F.pairwise_distance(train_out, test_out, keepdim=False).item()
            distances[label_idx] = distances.get(label_idx, []) + [distance]
        # minimum average distance corresponds to top score
        avg_distances = {key: sum(values) / len(values) for key, values in
                         distances.items()}
        top_k_indices = sorted(avg_distances, key=avg_distances.get)[:k]
        k_cat_labels = int_label_to_cat[top_k_indices]

        # if score falls below a certain threshold predict "new_whale"
        new_whale_inserted = False
        p = 0
        predictions[img_fname] = [""] * k
        for w in range(k):
            if new_whale_inserted or \
                    avg_distances[top_k_indices[p]] < threshold:
                predictions[img_fname][w] = k_cat_labels[p]
                p += 1
            else:
                predictions[img_fname][w] = "new_whale"
                new_whale_inserted = True
    return predictions


def create_submission_file(predictions: dict, save_path: str):
    """
    Create a submission file for kaggle competition
    :param predictions: dict, maps image filenames to list of predictions
    :param save_path: str, path to save submission csv to
    :return: pd.DataFrame, the submission
    """
    submission_df = pd.DataFrame(list(predictions.items()), columns=['Image', 'Id'])
    submission_df["Id"] = submission_df["Id"].map(lambda lst: " ".join(str(x) for x in lst))
    submission_df.to_csv(save_path, index=False)
    return submission_df


def main(device: str):
    torch.backends.cudnn.enabled = False
    print("Using device:", device)

    results_dir = "../results"
    model_name = "effnetb2_twinsiamese256x256"

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
    weights_path = os.path.join(results_dir, model_name, "model_weights.pth")
    n_features = 128
    backbone = efficientnet_b2(num_classes=n_features)
    model = BasicTwinSiamese(backbone)
    model.to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device(device)))

    train_outs, test_outs = get_siamese_backbone_outs(model, train_loader, test_loader, device)
    # get predictions and create submission file
    thresh = 2.0
    predictions = get_siamese_predictions(train_outs, test_outs,
                                          train_ds.int_label_to_cat, device, threshold=thresh)
    submission_path = os.path.join(results_dir, model_name, "test_submission_%.2f_thresh.csv" % thresh)
    create_submission_file(predictions, submission_path)
    print("Successfully saved submission file to:", submission_path)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch_device = "cuda:0"
    else:
        torch_device = "cpu"
    main(torch_device)
