from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

def get_min_wh(ds: Dataset) -> tuple:
    mapped = []
    img = False
    if type(ds[0][0]) == Image:
        img = True
    for i in range(len(ds)):
        if img:
            mapped.append(ds[i][0].size)
        else:
            mapped.append(ds[i][0].shape[1:])
            
    w = min(map(lambda x: x[0], mapped))
    h = min(map(lambda x: x[1], mapped))
    return w, h

def get_max_wh(ds: Dataset) -> tuple:
    mapped = []
    img = False
    if type(ds[0][0]) == Image:
        img = True
    for i in range(len(ds)):
        if img:
            mapped.append(ds[i][0].size)
        else:
            mapped.append(ds[i][0].shape[1:])
            
    w = max(map(lambda x: x[0], mapped))
    h = max(map(lambda x: x[1], mapped))
    return w, h

if __name__ == "__main__":
    from whale_dataset import WhaleDataset
    ds = WhaleDataset("../../data/train", "../../data/train.csv")
    print(get_min_wh(ds))
    print(get_max_wh(ds))