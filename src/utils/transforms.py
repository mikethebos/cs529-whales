from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

def get_min_hw(ds: Dataset) -> tuple:
    """
    Get min
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