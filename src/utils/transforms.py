import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


def get_blackwhite() -> transforms.Grayscale:
    """
    Convert all images to grayscale.
    :return out: transforms.Grayscale, transform for images
    """
    return transforms.Grayscale(1)


def get_centercrop(h: int, w: int) -> transforms.CenterCrop:
    """
    Crop edges of images such that they have specified dimensions.
    :param h: int, height to crop to
    :param w: int, width to crop to
    :return out: transforms.CenterCrop, transform for images
    """
    return transforms.CenterCrop((h, w))


def get_scale(h: int, w: int) -> transforms.Resize:
    """
    Scale image to specified dimensions.
    :param h: int, height to resize to
    :param w: int, width to resize to
    :return out: transforms.Resize, transform for images
    """
    return transforms.Resize((h, w))


def basic_transform(h: int, w: int, scale=False) -> transforms.Compose:
    """
    Makes images bb
    :param h: int, height to resize to
    :param w: int, width to resize to
    :param scale: bool, if true will resize images, false will crop
    :return out: transforms.Resize, transform for images
    """
    if scale:
        return transforms.Compose([get_blackwhite(), get_scale(h, w)])
    return transforms.Compose([get_blackwhite(), get_centercrop(h, w)])


def get_min_hw(ds: Dataset) -> tuple:
    """
    Get minimum height/width of images in dataset.
    :param ds: Dataset, the dataset of images
    :return out: tuple, (height, width)
    """
    mapped = []
    img = False
    if type(ds[0][0]) == Image:
        img = True
    for i in range(len(ds)):
        if img:
            mapped.append(ds[i][0].size)
        else:
            mapped.append(ds[i][0].shape[1:])

    h = min(map(lambda x: x[0], mapped))
    w = min(map(lambda x: x[1], mapped))
    return h, w


def get_max_hw(ds: Dataset) -> tuple:
    """
    Get maximum height/width of images in dataset.
    :param ds: Dataset, the dataset of images
    :return out: tuple, (height, width)
    """
    mapped = []
    img = False
    if type(ds[0][0]) == Image:
        img = True
    for i in range(len(ds)):
        if img:
            mapped.append(ds[i][0].size)
        else:
            mapped.append(ds[i][0].shape[1:])

    h = max(map(lambda x: x[0], mapped))
    w = max(map(lambda x: x[1], mapped))
    return h, w


def get_mean_std_of_channels(dl: Dataset, channels=1) -> tuple:
    means = []
    devs = []

    # get tensor of data; inspired by https://stackoverflow.com/a/74783382
    for ch in range(channels):
        means_for_ch = []
        means_sq = []
        for data in dl:
            imgs = data[0].cpu().numpy()
            means_for_ch.append(np.mean(imgs[ch, :, :]))
            means_sq.append(np.mean(imgs[ch, :, :] ** 2.0))
        means.append(np.mean(means_for_ch))
        devs.append(np.sqrt(np.mean(means_sq) - (means[-1] ** 2.0)))

    return means, devs


if __name__ == "__main__":
    from whale_dataset import WhaleDataset

    ds = WhaleDataset("../../data/train", "../../data/train.csv")

    h, w = 256, 256
    ds.transform = basic_transform(h, w, scale=True)
    means, devs = get_mean_std_of_channels(ds, channels=1)
    print("Means: ", means, ", devs: ", devs)
