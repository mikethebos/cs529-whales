"""
@author Jack Ringer, Mike Adams
Date: 5/1/2023
Description:
Contains code for data visualization/gathering stats
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.whale_dataset import WhaleDataset


def plot_images(images: list, images_per_row: int = 2, save_pth: str = None,
                show: bool = True):
    """
    Display multiple whale images. Based on show_whale() func from:
    https://www.kaggle.com/code/martinpiotte/whale-recognition-model-with-score-0-78563#Image-preprocessing

    :param images: list, images to show
    :param images_per_row: int, number of images per row in figure
    :param save_pth: str (optional), path to save fig to
    :param show: bool (optional), whether to show figure
    :return: None
    """
    n = len(images)
    rows = (n + images_per_row - 1) // images_per_row
    cols = min(images_per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(
        24 // images_per_row * cols, 24 // images_per_row * rows))
    for ax in axes.flatten():
        ax.axis('off')
    for i, (img, ax) in enumerate(zip(images, axes.flatten())):
        ax.imshow(img)
    plt.tight_layout()
    if save_pth is not None:
        plt.savefig(save_pth, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_class_frequencies(df: pd.DataFrame, save_pth: str = None,
                           show: bool = True):
    """
    Plot the frequency of each class in the data
    :param df: pd.DataFrame, from the train.csv file
    :param save_pth: str (optional), path to save fig to
    :param show: bool (optional), whether to show figure
    :return: None
    """
    class_counts = df['Id'].value_counts()
    plt.hist(class_counts.values, bins=50, log=True)
    plt.xlabel("Number of Instances")
    plt.ylabel('Frequency (log-scale)')
    plt.title('Distribution of Class Frequencies')
    if save_pth is not None:
        plt.savefig(save_pth, dpi=300)
    if show:
        plt.show()
    plt.close()


def main():
    ds = WhaleDataset("../data/train", "../data/train.csv")
    n_classes = len(ds.int_label_to_cat)
    print("Num classes:", n_classes)
    print("Num instances:", len(ds))
    # calculated from transforms
    print("Min, Max Height: %s, %s" % (30, 1613))
    print("Min, Max Width: %s, %s" % (64, 1050))
    n_show = 8
    ims = []
    for i in range(n_show):
        ims.append(ds[i][0])
    plot_images(ims, 4, save_pth="../figures/sample_images.png", show=False)
    plot_class_frequencies(ds.df, save_pth="../figures/class_distribution.png",
                           show=False)


if __name__ == "__main__":
    main()
