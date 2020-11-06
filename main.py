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
from dataloader import write
from utils import n_fold_split, knn_split_valid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)


print("Loading Training Data:")
train_y_tl = read_label(TRAIN_DIR)
train_x_tl = read_img(TRAIN_DIR, train=True)
print("Loading Testing Data:")
test_x = read_img(VALID_DIR, train=False)
train_mean, train_var, _, _, _ = zip(*train_y_tl)
data_size_tl = len(train_x_tl)
train_ids, valid_ids = \
    knn_split_valid() if USE_KNN_VALID else n_fold_split(data_size_tl, _val_fold_id=VAL_FOLD_ID)

train_x = train_x_tl[train_ids]
valid_x = train_x_tl[valid_ids]
del train_x_tl

train_mean = np.array(train_mean)
train_var = np.array(train_var)
valid_mean = train_mean[valid_ids]
train_mean = train_mean[train_ids]
valid_var = train_var[valid_ids]
train_var = train_var[train_ids]

print (f"Number of train: {len(train_x)}")
print (f"Number of Validate: {len(valid_x)}")
print (f"Number of test: {len(test_x)}")

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = ImgDataset(train_x, train_mean, train_var, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_set = ImgDataset(valid_x, valid_mean, valid_var, transform=train_transform)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
train_size = len(train_set)
model = FoodNet().to(DEVICE)

if DISCRETE_CLS:
    mean_loss, var_loss = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
else:
    mean_loss, var_loss = nn.MSELoss(), nn.MSELoss()

if MEAN_MODEL_NAME == "cnn" and not SEPARATE:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

elif MEAN_MODEL_NAME == "resnet" and not SEPARATE:
    # Non-Separated ResNet
    optimizer = torch.optim.Adam([{'params': model.net.mlp.parameters(), 'lr': NEW_PARAM_LR},
                                  {'params': model.net.mlp_var.parameters(), 'lr': NEW_PARAM_LR},
                                  {'params': model.net.resnet.parameters(), 'lr': FINE_TUNE_LR}], weight_decay=WEIGHT_DECAY)

elif MEAN_MODEL_NAME == "resnet" and VAR_MODEL_NAME == "cnn" and SEPARATE:
    # Non-Separated ResNet
    optimizer = torch.optim.Adam([{'params': model.mean_net.resnet.parameters(), 'lr': FINE_TUNE_LR},
                                  {'params': model.mean_net.mlp.parameters(), 'lr': NEW_PARAM_LR},
                                  {'params': model.var_net.parameters(), 'lr': NEW_PARAM_LR}], weight_decay=WEIGHT_DECAY)
else:
    optimizer = None

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

best_result = 1e2
converge_epoch_count = 0
for epoch in range(EPOCH_NUM):
    rmse_mean, rmse_var, loss, loss_mean, loss_var = run_epoch(model, train_loader, "train", optimizer, mean_loss, mean_loss)
    valid_rmse_mean, valid_rmse_var = run_epoch(model, valid_loader, "val")
    valid_rmse = 0.6 * valid_rmse_mean + 0.4 * valid_rmse_var
    lr_scheduler.step()

    rmse_diff = best_result - valid_rmse
    if rmse_diff > CONVERGE_INCREMENT_THRESHOLD:
        best_result = valid_rmse
        torch.save(model, BEST_MODEL_PATH)
        print("Best Model Saved at", BEST_MODEL_PATH)
        converge_epoch_count = 0
    else:
        converge_epoch_count += 1

    print (f"Epoch: {epoch}; train loss: {round(loss / train_size, ROUND_NUM)}, "
           f"Best rmse: {round(best_result.item(), ROUND_NUM)}; "
           f"val_rmse_mean: {round(valid_rmse_mean.item(), ROUND_NUM)}; "
           f"val_rmse_var: {round(valid_rmse_var.item(), ROUND_NUM)}; "
           f"val_rmse: {round(valid_rmse.item(), ROUND_NUM)}; ")

    if converge_epoch_count > CONVERGE_EPOCH_NUM_THRESHOLD:
        print("Early Stop Happened.")
        break

if USE_FULL_DATA:
    best_model = model
else:
    del model
    best_model = torch.load(BEST_MODEL_PATH)

# fine-tune
if not USE_FULL_DATA and FINE_TUNE_EPOCH > 0:
    for epoch in range(FINE_TUNE_EPOCH):
        rmse_mean, rmse_var, loss, loss_mean, loss_var = \
            run_epoch(best_model, valid_loader, "train", optimizer, mean_loss, mean_loss)

with torch.no_grad():
    test_pred_mean, test_pred_var = run_epoch(best_model, test_loader, "test")
    test_pred_mean = test_pred_mean.reshape(-1,).tolist()
    test_pred_var = test_pred_var.reshape(-1).tolist()

if TOY:
    exit()

write(test_pred_mean, test_pred_var, RES_PATH)
