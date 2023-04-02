from torch.utils.data import Dataset
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

def basic_transform(h: int, w: int, upscale=False) -> transforms.Compose:
    """
    Makes images bb
    :param h: int, height to resize to
    :param w: int, width to resize to
    :return out: transforms.Resize, transform for images
    """
    if upscale:
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

if __name__ == "__main__":
    from whale_dataset import WhaleDataset
    ds = WhaleDataset("../../data/train", "../../data/train.csv")
    print(get_min_hw(ds))
    print(get_max_hw(ds))