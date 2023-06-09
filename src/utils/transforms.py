import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms


def _toRGB_wrapper(image, **kwargs):
    if A.is_rgb_image(image):
        return image
    return A.ToRGB(p=1.0)(image)


def train_alb_transform(resize_height: int, resize_width: int,
                        channel_means: list, channel_stds: list,
                        toRGB: bool = False):
    """
    Transformations used for training data
    :param resize_height: int, height to resize images to
    :param resize_width: int, width to resize images to
    :param channel_means: list[float], mean values of image channels
    :param channel_stds: list[float], standard deviations of image channels
    :param toRGB: bool, true if all images should be cast to RGB
    :return: albumentations transform
    """
    col = A.ToGray(p=1.0)
    if toRGB:
        col = A.Lambda(_toRGB_wrapper, p=1.0)
    tf = A.Compose([
        A.Affine(rotate=(-10, 10), translate_percent=(0.0, 0.05), shear=(-2, 2),
                 p=0.5),
        A.Resize(height=resize_height, width=resize_width),
        A.GaussianBlur(blur_limit=(3, 7), p=0.05),
        A.GaussNoise(p=0.05),
        A.Posterize(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomSnow(p=0.1),
        A.RandomRain(p=0.05),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=channel_means, std=channel_stds),
        col,
        ToTensorV2()
    ])
    return tf


def test_alb_transform(resize_height: int, resize_width: int,
                       channel_means: list, channel_stds: list,
                       toRGB: bool = False):
    """
    Basic transform using albumentations library. Used for model
    inference.
    :param resize_height: int, height to resize images to
    :param resize_width: int, width to resize images to
    :param channel_means: list[float], mean values of image channels
    :param channel_stds: list[float], standard deviations of image channels
    :param toRGB: bool, true if all images should be cast to RGB
    :return: albumentations transform
    """
    col = A.ToGray(p=1.0)
    if toRGB:
        col = A.Lambda(_toRGB_wrapper, p=1.0)
    tf = A.Compose([
        A.Resize(height=resize_height, width=resize_width),
        A.Normalize(mean=channel_means, std=channel_stds),
        col,
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
    import torch
    import matplotlib.pyplot as plt
    from whale_dataset import WhaleDataset
    from torch.utils.data import random_split

    h, w = 256, 256
    means = [140.1891, 147.7153, 156.5466]
    stds = [71.8512, 68.4338, 67.5585]
    # normalize the means and stds for albumentations usage
    means = [m / 255.0 for m in means]
    stds = [s / 255.0 for s in stds]

    tf = train_alb_transform(h, w, means, stds)
    dataset = WhaleDataset("../../data/train", "../../data/train.csv",
                           transform=None)
    ds_train, ds_val = random_split(dataset, [int(len(dataset) * 0.8),
                                              int(len(dataset) * 0.2)])
    ds_train.dataset.transform = tf
    img, label = ds_train[67]
    img = img.permute(1, 2, 0)
    plt.imshow(img, vmin=torch.min(img).item(), vmax=torch.max(img).item())
    plt.show()
