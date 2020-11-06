import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from const import *
from progressbar import ProgressBar
from utils import discrete_label
from utils import n_fold_split


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


class ImgDataset2(Dataset):
    def __init__(self, filepath, transform, train=True, valid=False):
        self.train = train
        self.valid = valid
        self.img = read_img(filepath, train=train)
        self.transform = transform

        if self.train:
            self.mean_y, self.var_y, _, _, _ = zip(*read_label(filepath))
            self.mean_y = np.array(self.mean_y)
            self.var_y = np.array(self.var_y)

        if self.valid:
            train_ids, valid_ids = n_fold_split(len(self.img))
            self.img = self.img[valid_ids]
            self.mean_y = self.mean_y[valid_ids]
            self.var_y = self.var_y[valid_ids]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.transform(self.img[index])
        if not self.train:
            return img

        mean_y = self.mean_y[index]
        var_y = self.var_y[index]
        return img, mean_y, var_y


def read_img(path, train=False):
    data_size = 3000 if train else 1500
    image_dir = ["%s.jpg" % i for i in range(data_size)]
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
    mean, var, _, _, _ = zip(*y)
    return y


def knn_split(train, test):
    def dist(sample, batch):
        batch_size = batch.shape[0]
        d = np.abs(batch - sample).reshape(batch_size, -1).mean(axis=1)
        return d
    img_trn = read_img(train, train=True)
    img_tst = read_img(test, train=False)
    trn_size, tst_size = len(img_trn), len(img_tst)
    valid_score = {}
    _bar = ProgressBar(max_value=tst_size)
    for i in range(tst_size):
        _bar.update(i+1)
        dists = dist(img_tst[i], img_trn)
        best_j = dists.argmin()
        if best_j not in valid_score:
            valid_score[best_j] = 1e10
        valid_score[best_j] = dists[best_j] if dists[best_j] < valid_score[best_j] else valid_score[best_j]
    top = [k for k in sorted(valid_score, key=valid_score.get)]
    valid_id = top[:trn_size//10]
    train_id = [i for i in range(trn_size) if i not in valid_id]
    return train_id, valid_id


if __name__ == "__main__":
    train_ids, valid_ids = knn_split(TRAIN_DIR, VALID_DIR)
    print(train_ids)
    print(valid_ids)

