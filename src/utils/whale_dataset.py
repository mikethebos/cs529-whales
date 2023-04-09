"""
@author Mike Adams, Jack Ringer
Date: 3/29/2023
Description: Class used to load images and labels for the happy whale dataset
using PyTorch Dataset.
"""
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import random
import os


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
        # force into RGB to make Grayscale transform work
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.int_labels.iloc[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image.float(), torch.tensor(label).long()

    def get_cat_for_label(self, int_label):
        return self.int_label_to_cat[int_label]


class SiameseDataset(Dataset):
    def __init__(self, images_dir: str, csv_file: str, transform=None):
        """
        Initialize dataset for use with Siamese Network.
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

    def __len__(self):
        return len(self.df)  # may be len(self.df) ** 2

    def __getitem__(self, idx):
        label1 = self.df["Id"].sample(n=1).iloc[0]
        use_same_class = random.randint(0, 1)
        img1_fname, img2_fname = "", ""
        if use_same_class:
            while len(self.id_files_map[label1]) <= 1:
                # need at least two images
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
        img1 = read_image(img1_path, ImageReadMode.RGB)
        img2 = read_image(img2_path, ImageReadMode.RGB)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1.float(), img2.float(), torch.tensor(use_same_class).long()

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
        # force into RGB to make Grayscale transform work
        image = read_image(img_path, ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image.float()


def plot_img(image):
    plt.imshow(image[0, :, :], cmap='gray')
    plt.show()


def plot_images_siamese(image1, image2, are_same):
    fig, ax = plt.subplots(1, 2)

    # Display the images side-by-side in the subplots
    ax[0].imshow(image1[0, :, :], cmap='gray')
    ax[0].set_title("Image 1")
    ax[1].imshow(image2[0, :, :], cmap='gray')
    ax[1].set_title("Image 2")
    fig.suptitle("Same? " + str(are_same))
    plt.show()


if __name__ == "__main__":
    from src.utils.transforms import basic_transform

    tf = basic_transform(256, 256, True)
    ds = SiameseDataset("../../data/train", "../../data/train.csv",
                        transform=tf)
    dl = torch.utils.data.DataLoader(ds,
                                     shuffle=True,
                                     batch_size=4)
    # plot_img_siamese(im1, im2, same)
    ex_batch = next(iter(dl))
    im1s, im2s, same = ex_batch
    for im1, im2, s in zip(im1s, im2s, same):
        plot_images_siamese(im1, im2, s)
