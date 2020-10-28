from const import *
import torch
import numpy as np


def rmse(pred, gold):
    score = torch.sqrt(((pred - gold)**2).mean())
    return score


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
