#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from const import *
from models import run_epoch
from models import FoodNet
from dataloader import ImgDataset
from dataloader import read_img, read_label
from dataloader import n_fold_split
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


print("Loading Training Data:")
_train_x = read_img(train_dir)
_train_y = read_label(train_dir)
print("Loading Testing Data:")
test_x = read_img(test_dir)
train_mean, train_var, _, _, _ = zip(*_train_y)
train_ids, valid_ids = n_fold_split(len(_train_x[0]), _val_fold_id=val_fold_id)

train_x = [c[train_ids] for c in _train_x]
valid_x = [c[valid_ids] for c in _train_x]
del _train_x

train_mean = np.array(train_mean)
train_var = np.array(train_var)
valid_mean = train_mean[valid_ids]
train_mean = train_mean[train_ids]
valid_var = train_var[valid_ids]
train_var = train_var[train_ids]

print (f"Number of train: {[len(i) for i in train_x]}")
print (f"Number of Validate: {[len(i) for i in valid_x]}")
print (f"Number of test: {[len(i) for i in test_x]}")

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = ImgDataset(train_x, train_mean, train_var, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_set = ImgDataset(valid_x, valid_mean, valid_var, transform=train_transform)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
train_size = len(train_set)
model = FoodNet().to(device)

if discrete_cls:
    mean_loss, var_loss = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
else:
    mean_loss, var_loss = nn.MSELoss(), nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Non-Separated ResNet
optimizer = torch.optim.Adam([{'params': model.net.mlp.parameters(), 'lr': 1e-3},
                              {'params': model.net.mlp_var.parameters(), 'lr': 1e-3},
                              {'params': model.net.resnet.parameters(), 'lr': 1e-4}], weight_decay=5e-4)
best_result = 1e2

converge_epoch_count = 0
for epoch in range(num_epoch):
    rmse_mean, rmse_var, loss, loss_mean, loss_var, optimizer = \
        run_epoch(model, train_loader, "train", optimizer, mean_loss, mean_loss)
    valid_rmse_mean, valid_rmse_var = run_epoch(model, valid_loader, "val")
    valid_rmse = 0.6 * valid_rmse_mean + 0.4 * valid_rmse_var

    rmse_diff = best_result - valid_rmse
    if rmse_diff > CONVERGE_INCREMENT_THRESHOLD:
        best_result = valid_rmse
        torch.save(model, best_model_path)
        print("Best Model Saved at", best_model_path)
        converge_epoch_count = 0
    else:
        converge_epoch_count += 1

    print (f"Epoch: {epoch}; train loss: {round(loss/train_size, round_num)}, "
           f"Best rmse: {round(best_result.item(), round_num)}; "
           f"val_rmse_mean: {round(valid_rmse_mean.item(), round_num)}; "
           f"val_rmse_var: {round(valid_rmse_var.item(), round_num)}; "
           f"val_rmse: {round(valid_rmse.item(), round_num)}; ")

    if converge_epoch_count > CONVERGE_EPOCH_NUM_THRESHOLD:
        print("Early Stop Happened.")
        break

del model
best_model = torch.load(best_model_path)
best_model.eval()
with torch.no_grad():
    test_pred_mean, test_pred_var = run_epoch(best_model, test_loader, "test")
    test_pred_mean = test_pred_mean.reshape(-1,).tolist()
    test_pred_var = test_pred_var.reshape(-1).tolist()

if toy:
    exit()

with open(res_path, 'w') as f:
    for z1, z2 in zip(test_pred_mean, test_pred_var):
        print (round(z1, round_num), round(z2, round_num))
        f.write(str(round(z1, round_num))+'\t'+str(round(z2, round_num))+"\n")






