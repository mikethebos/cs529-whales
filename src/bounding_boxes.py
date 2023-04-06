"""
@author Jack Ringer, Mike Adams
Date: 4/5/2023
Description:
Script used to display bounding boxes for different images from training/testing
Acknowledgement:
Much of this code is based on:
https://www.kaggle.com/code/martinpiotte/bounding-box-data-for-the-whale-flukes/notebook

Bounding box data is from:
https://www.kaggle.com/datasets/martinpiotte/humpback-whale-identification-fluke-location
"""
from typing import List
import os
import pandas as pd
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_bounding_box_df(bbox_file: str):
    """
    Load bounding box data into Pandas DataFrame
    :param bbox_file: str, path to bounding box data (filename, coordinates)
    :return: pd.DataFrame mapping filename to list of coordinates for bbox
    """
    with open(bbox_file) as f:
        bbox_data = f.read().split('\n')
    bbox_data = [line.split(',') for line in bbox_data]
    bbox_data = [(p, [(int(coord[i]), int(coord[i + 1])) for i in
                      range(0, len(coord), 2)]) for p, *coord in bbox_data]
    # bbox_data[i] = (fname, list[(x,y)])
    bbox_df = pd.DataFrame(bbox_data, columns=['Image', 'coords'])
    return bbox_df


def get_bounding_box(coordinates: List[tuple]):
    """
    Get the bounding box for a given list of coordinates
    :param coordinates: list[tuple], list of 2-tuple (x,y)
    :return: 4-tuple: x_min, y_min, x_max, y_max
    """
    if len(coordinates) <= 0:
        return None
    x_min, y_min = coordinates[0]
    x_max, y_max = x_min, y_min
    for x, y in coordinates[1:]:
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    return x_min, y_min, x_max, y_max


def display_bbox(img: torch.Tensor, box: tuple):
    """
    Display bounding box over image using matplotlib
    :param img: torch.Tensor [H,W,C]
    :param box: 4-tuple containing x_min, y_min, x_max, y_max
    :return: None
    """
    x_min, y_min, x_max, y_max = box
    # show image and bounding box
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    plt.show()


def main(img_idx: int):
    """
    Script to load and display a bounding box over a training image
    :param img_idx: int, index from bbox_df to show
    :return: None
    """
    bbox_df = load_bounding_box_df("../data/bboxes.txt")
    test_file = bbox_df['Image'].iloc[img_idx]
    test_coords = bbox_df.loc[bbox_df['Image'] == test_file]['coords'].iloc[0]
    box = get_bounding_box(test_coords)
    test_img = read_image(os.path.join("../data/train", test_file))
    test_img = test_img.permute(1, 2, 0)  # channels last for matplotlib
    display_bbox(test_img, box)


if __name__ == "__main__":
    main(60)
