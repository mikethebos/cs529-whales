import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms


def basic_alb_transform(resize_height: int, resize_width: int,
                        channel_means: list, channel_stds: list):
    """
    Basic transform using albumentations library
    :param resize_height: int, height to resize images to
    :param resize_width: int, width to resize images to
    :param channel_means: list[float], mean values of image channels
    :param channel_stds: list[float], standard deviations of image channels
    :return: albumentations transform
    """
    tf = A.Compose([
        A.Resize(height=resize_height, width=resize_width),
        A.Normalize(mean=channel_means, std=channel_stds),
        A.ToGray(p=1.0),
        ToTensorV2()
    ])
    return tf


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
    from whale_dataset import WhaleDataset, plot_img
    from torch.utils.data import random_split

    h, w = 256, 256
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]

    tf = basic_alb_transform(h, w, means, stds)
    dataset = WhaleDataset("../../data/train", "../../data/train.csv",
                           transform=tf)
    ds_train, ds_val = random_split(dataset, [int(len(dataset) * 0.8),
                                              int(len(dataset) * 0.2)])
    img, label = ds_val[10]
    print(img.shape)
    img = img.permute(1, 2, 0)  # plotting expects (H, W, C) not (C, H, W)
    plot_img(img)
