"""
@author Mike Adams, Jack Ringer
Date: 3/29/2023
Description: Class used to load images and labels for the happy whale dataset
using PyTorch Dataset.
"""
import os
import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


def read_image(image_path: str):
    """
    Read in an image using the cv2 library (compatible with Albumentations)
    :param image_path: str, path to image
    :return: numpy.ndarray representing image <H, W, C=3>
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class WhaleDataset(Dataset):
    def __init__(self, images_dir: str, csv_file: str, transform=None):
        """
        Initialize whale dataset for use with Torch models.
        :param images_dir: str, path to image directory
        :param csv_file: str, path to csv mapping image filenames to labels
                            (1 col image names, 1 col labels)
        :param transform, a callable to apply to each image tensor
        :return: None
        """
        self.img_dir = images_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        cats = self.df["Id"].astype("category").cat
        # map unique str labels to unique int id
        self.int_labels = cats.codes
        self.int_label_to_cat = cats.categories

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_fname = self.df["Image"].iloc[idx]
        img_path = os.path.join(self.img_dir, img_fname)
        image = read_image(img_path)
        label = self.int_labels.iloc[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label).long()

    def get_cat_for_label(self, int_label):
        return self.int_label_to_cat[int_label]


class TwinSiameseDataset(Dataset):
    def __init__(self, images_dir: str, csv_file: str, transform=None):
        """
        Initialize dataset for use with Twin Siamese Network.
        :param images_dir: str, path to image directory
        :param csv_file: str, path to csv mapping image filenames to labels
                            (1 col image names, 1 col labels)
        :param transform, a callable to apply to each image tensor
        :return: None
        """
        self.img_dir = images_dir
        self.df = pd.read_csv(csv_file)
        # map ids to list of filenames w/ that label
        self.id_files_map = self.df.groupby('Id')['Image'].apply(list).to_dict()
        self.transform = transform
        cats = self.df["Id"].astype("category").cat
        # map unique str labels to unique int id
        self.int_labels = cats.codes
        self.int_label_to_cat = cats.categories

    def __init__(self, subset: Subset):
        """
        Initialize dataset for use with Twin Siamese Network with a subset of data.
        :param subset: Subset, subset of a whale dataset
        :return: None
        """
        self.img_dir = subset.dataset.img_dir
        self.df = subset.dataset.df
        self.transform = subset.dataset.transform
        self.int_labels = subset.dataset.int_labels
        self.int_label_to_cat = subset.dataset.int_label_to_cat

        self.df = self.df.iloc[subset.indices]
        self.id_files_map = self.df.groupby('Id')['Image'].apply(list).to_dict()
        self.int_labels = self.int_labels.iloc[subset.indices]

    def __len__(self):
        return len(self.df)  # may be len(self.df) ** 2

    def __getitem__(self, idx):
        label1 = self.df["Id"].sample(n=1).iloc[0]
        use_same_class = random.randint(0, 1)
        img1_fname, img2_fname = "", ""
        if use_same_class:
            while label1 == "new_whale" or len(self.id_files_map[label1]) <= 1:
                # new_whale not really same class and need at least two images
                label1 = self.df["Id"].sample(n=1).iloc[0]
            candidate_fnames = self.id_files_map[label1]
            img1_fname, img2_fname = random.sample(candidate_fnames, k=2)
        else:
            img1_fname = self.df["Image"].iloc[idx]
            label2 = self.df["Id"].sample(n=1).iloc[0]
            while label2 == label1:
                # want labels to be different
                label2 = self.df["Id"].sample(n=1).iloc[0]
            img2_candidates = self.id_files_map[label2]
            img2_fname = random.choice(img2_candidates)
        img1_path = os.path.join(self.img_dir, img1_fname)
        img2_path = os.path.join(self.img_dir, img2_fname)
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        if self.transform is not None:
            img1 = self.transform(image=img1)["image"]
            img2 = self.transform(image=img2)["image"]
        # contrastive loss expects negative as 1, positive as 0
        label = not use_same_class
        return img1, img2, torch.tensor(label).long()

    def get_cat_for_label(self, int_label: int):
        return self.int_label_to_cat[int_label]


class TestWhaleDataset(Dataset):
    def __init__(self, images_dir: str, transform=None):
        """
        Initialize test dataset (data with no labels).
        :param images_dir: str, path to image directory
        :param transform, a callable to apply to each image tensor
        :return: None
        """
        self.img_dir = images_dir
        self.images = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_fname = self.images[idx]
        img_path = os.path.join(self.img_dir, img_fname)
        image = read_image(img_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, img_fname


def plot_img(image):
    plt.imshow(image)
    plt.show()


def plot_images_siamese(image1, image2, are_same):
    fig, ax = plt.subplots(1, 2)

    # Display the images side-by-side in the subplots
    ax[0].imshow(image1[0, :, :], cmap='gray')
    ax[0].set_title("Image 1")
    ax[1].imshow(image2[0, :, :], cmap='gray')
    ax[1].set_title("Image 2")
    fig.suptitle("Different? " + str(are_same))
    plt.show()


if __name__ == "__main__":
    import numpy as np
    ds = WhaleDataset("../../data/train", "../../data/train.csv")
    n_classes = len(ds.int_label_to_cat)
    scale = (2 ** 0.5) * np.log(n_classes - 1)
    print(scale)
    print(1 / n_classes)