"""
@author Mike Adams, Jack Ringer
Date: 3/29/2023
Description: Class used to load images and labels for the happy whale dataset
using PyTorch Dataset.
"""
import matplotlib.pyplot as plt  # temporary

from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
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
        image = read_image(img_path)
        label = self.int_labels.iloc[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def get_cat_for_label(self, int_label):
        return self.int_label_to_cat[int_label]


def plot_img(image):
    plt.imshow(image[0, :, :], cmap='gray')
    plt.show()


if __name__ == "__main__":
    ds = WhaleDataset("../../data/train", "../../data/train.csv")
    print(len(ds.df[ds.df["Id"] == 'new_whale']))  # decent number of new whales
    img, img_label = ds[0]
    print(img_label)
    print("verify cat_label ", ds.get_cat_for_label(img_label))
    plot_img(img)