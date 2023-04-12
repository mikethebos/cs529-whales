"""
@author Jack Ringer, Mike Adams
Date: 4/11/2023
Description:
Simple script to calculate the means/standard deviation of images from a folder
"""
from utils.whale_dataset import TestWhaleDataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def main(dir_path: str):
    """
    Calculate mean and standard deviation of images in given directory
    :param dir_path: str, path to directory
    :return: channel means and standard deviations
    """
    tf = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])

    dataset = TestWhaleDataset(images_dir=dir_path, transform=tf)
    n = len(dataset)
    images = torch.zeros((n, 3, 256, 256))
    for i in tqdm(range(n)):
        images[i] = dataset[i]
    means = torch.mean(images, dim=(0, 2, 3))
    stds = torch.std(images, dim=(0, 2, 3))
    print("Mean:", means)  # tensor([140.1891, 147.7153, 156.5466])
    print("Std:", stds)  # tensor([71.8512, 68.4338, 67.5585])
    return means, stds


if __name__ == "__main__":
    main("../data/train")
