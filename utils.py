import torch
import numpy as np
import random
from progressbar import ProgressBar
from const import *


def rmse(pred, gold):
    score = torch.sqrt(((pred - gold)**2).mean())
    return score


def n_fold_split(_data_size, _val_fold_id=0, _fold_num=10):
    fold_size = _data_size // _fold_num
    ids = list(range(_data_size))
    random.seed(2020)
    random.shuffle(ids)
    st, ed = 0, _data_size
    st_val = _val_fold_id * fold_size
    ed_val = st_val + fold_size
    valid = ids[st_val:ed_val]
    train = ids if USE_FULL_DATA else ids[st:st_val] + ids[ed_val:ed]
    return train, valid


def knn_split(img_trn, img_tst):
    def dist(sample, batch):
        batch_size = batch.shape[0]
        d = np.abs(batch - sample).reshape(batch_size, -1).mean(axis=1)
        return d
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


def knn_split_valid():
    valid_id = VALID_IDS_KNN
    train_id = [i for i in range(3000) if i not in valid_id]
    return train_id, valid_id


def discrete_label(y, cat="mean"):
    assert cat in ["mean", "var"]
    cls_map = discrete_mean_map if cat == "mean" else discrete_var_map
    _round_num = 2 if cat == "mean" else 4
    discrete_y = torch.tensor(cls_map[round(y.item(), _round_num)], dtype=torch.long)
    return discrete_y


def un_discrete_label(discrete_y, cat="mean", gold=False):
    if not discrete_y:
        return discrete_y

    assert cat in ["mean", "var"]
    value_map = discrete_means if cat == "mean" else discrete_vars
    if not gold:
        discrete_y = torch.argmax(discrete_y, dim=1)
    y = torch.zeros_like(discrete_y, dtype=torch.float)

    for i in range(discrete_y.shape[0]):
        y[i] = value_map[discrete_y[i].item()]
    return y


def post_process(mean, var):
    def nearest(neighbors: [int], e: int):
        d = np.array(neighbors) - e
        d = np.abs(d)
        i = np.argmin(d)
        return neighbors[i]
    m = torch.zeros_like(mean)
    v = torch.zeros_like(var)
    for _i, [_m, _v] in enumerate(zip(mean, var)):
        m[_i] = nearest(discrete_means, _m.item())
        v[_i] = nearest(discrete_vars, _v.item())
    return m, v


