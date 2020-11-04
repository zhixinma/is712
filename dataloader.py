import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
from const import *
from progressbar import ProgressBar
from utils import discrete_label


class ImgDataset(Dataset):
    def __init__(self, x, mean_y=None, var_y=None, transform=None):
        self.img = x
        self.mean_y = mean_y
        self.var_y = var_y
        if mean_y is not None:
            self.mean_y = torch.tensor(mean_y, dtype=torch.float)
            self.var_y = torch.tensor(var_y, dtype=torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.img[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.mean_y is not None:
            mean_y = self.mean_y[index]
            var_y = self.var_y[index]
            if DISCRETE_CLS:
                mean_y = discrete_label(mean_y, "mean")
                var_y = discrete_label(var_y, "var")
            return img, mean_y, var_y
        else:
            return img


def read_img(path):
    image_dir = sorted([file for file in os.listdir(path) if "txt" not in file])
    image_dir = image_dir[:20] if TOY else image_dir
    _data_size = len(image_dir)
    images = np.zeros((_data_size, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]), dtype=np.uint8)
    _bar = ProgressBar(max_value=_data_size)
    for img_i, file in enumerate(image_dir):
        _bar.update(img_i+1)
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(img, IMG_SHAPE[:2])
        images[img_i] = img
    print()
    return images


def read_label(path):
    with open(os.path.join(path, "label_train.txt"), 'r') as f:
        lines = f.readlines()
    y = [[float(e) for e in line.split('\t')] for line in lines]
    return y


def n_fold_split(_data_size, _val_fold_id=0, _fold_num=10):
    fold_size = _data_size // _fold_num
    ids = list(range(_data_size))
    random.seed(2020)
    random.shuffle(ids)
    st, ed = 0, _data_size
    st_val = _val_fold_id * fold_size
    ed_val = st_val + fold_size
    valid = ids[st_val:ed_val]
    train = ids[st:st_val] + ids[ed_val:ed]
    # train = ids
    return train, valid

