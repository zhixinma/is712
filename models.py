import torch
import torch.nn as nn
import torchvision
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA
from const import *
from utils import rmse
from utils import un_discrete_label, post_process


class FoodNet(nn.Module):
    def __init__(self):
        super(FoodNet, self).__init__()
        model_dict = {"resnet": ResNet, "cnn": CNN}
        assert MEAN_MODEL_NAME in model_dict, "Mean:%s" % MEAN_MODEL_NAME
        if SEPARATE:
            assert VAR_MODEL_NAME in model_dict, "Var:%s" % VAR_MODEL_NAME
            self.mean_net = model_dict[MEAN_MODEL_NAME]()
            self.var_net = model_dict[VAR_MODEL_NAME]()
        else:
            self.net = model_dict[MEAN_MODEL_NAME](SEPARATE)

    def forward(self, img):
        if SEPARATE:
            mean_out = self.mean_net(img)
            var_out = self.var_net(img)
        else:
            mean_out, var_out = self.net(img)

        if DISCRETE_CLS:
            pass  # Cross Entropy Loss includes softmax function.
        return mean_out, var_out


class ResNet(nn.Module):
    def __init__(self, _separate=True):
        super(ResNet, self).__init__()
        self.separate = _separate
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        for param in self.resnet.parameters():
            param.reguires_grad = UPDATE_RESNET

        d_in = 512
        d_out_mean = MEAN_CLS_NUM if DISCRETE_CLS else 1
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out_mean))

        if not self.separate:
            d_out_var = VAR_CLS_NUM if DISCRETE_CLS else 1
            self.mlp_var = nn.Sequential(
                nn.Linear(d_in, d_out_var))

    def forward(self, img):
        output = self.resnet.conv1(img)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)
        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)
        output = self.resnet.layer4(output)
        output = self.resnet.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.dropout(output)

        res = self.mlp(output)
        if self.separate:
            return res

        mean = res
        var = self.mlp_var(output)
        return mean, var


class CNN(nn.Module):
    def __init__(self, _separate=True):
        super(CNN, self).__init__()
        self.separate = _separate
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(3)
        self.max_pool2 = nn.MaxPool2d(3)
        self.max_pool3 = nn.MaxPool2d(3)
        self.avg_pool = nn.AvgPool2d(3)
        self.dropout = nn.Dropout(p=0.5)

        d_in = 256
        d_out_mean = MEAN_CLS_NUM if DISCRETE_CLS else 1
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, d_out_mean))

        if not self.separate:
            d_out_var = VAR_CLS_NUM if DISCRETE_CLS else 1
            self.mlp_var = nn.Sequential(
                nn.Linear(d_in, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, d_out_var))

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.max_pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.max_pool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.max_pool3(output)
        output = self.avg_pool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.dropout(output)

        res = self.mlp(output)
        if self.separate:
            return res

        mean = res
        var = self.mlp_var(output)
        return mean, var


def run_epoch(_model, data_loader, tag, optimizer=None, mean_loss=None, var_loss=None):
    assert tag in ["train", "val", "test"], "tag should be train, val or test"
    _loss, _loss_mean, _loss_var = 0, 0, 0
    mean_pred, var_pred, mean_gold, var_gold = [], [], [], []
    mean_gold_batch, var_gold_batch = None, None

    if tag == 'train':
        _model.train()
    else:
        _model.eval()

    data_size = len(data_loader.dataset)
    widgets = [tag, Percentage(), ' ', Bar('-'), ' ', Timer(), ' ', ETA()]
    _bar = ProgressBar(widgets=widgets, max_value=data_size)
    t = 0
    for i, batch_data in enumerate(data_loader):
        if tag == "train" or tag == "val":
            _image_idx, _mean_idx, _var_idx = 0, 1, 2
            image_batch = batch_data[_image_idx].to(DEVICE)
            mean_gold_batch = batch_data[_mean_idx].unsqueeze(-1)
            var_gold_batch = batch_data[_var_idx].unsqueeze(-1)
            mean_gold.append(mean_gold_batch)
            var_gold.append(var_gold_batch)
            mean_gold_batch = mean_gold_batch.to(DEVICE)
            var_gold_batch = var_gold_batch.to(DEVICE)

        else:
            image_batch = batch_data.to(DEVICE)

        t += image_batch.shape[0]
        _bar.update(t)

        if tag == "train":
            optimizer.zero_grad()

        mean_pred_batch, var_pred_batch = _model(image_batch)
        mean_pred.append(mean_pred_batch.cpu().detach())
        var_pred.append(var_pred_batch.cpu().detach())

        if DISCRETE_CLS and tag == "train":
            mean_gold_batch = mean_gold_batch.reshape(-1)
            var_gold_batch = var_gold_batch.reshape(-1)

        if tag == "train":
            batch_mean_loss = mean_loss(mean_pred_batch, mean_gold_batch)
            batch_var_loss = var_loss(var_pred_batch, var_gold_batch)
            batch_loss = 0.6 * batch_mean_loss + 0.4 * batch_var_loss
            batch_loss.backward()
            optimizer.step()
            _loss += batch_loss.item()
            _loss_mean += batch_mean_loss.item()
            _loss_var += batch_var_loss.item()
    print()

    mean_pred = torch.cat(mean_pred, dim=0)
    var_pred = torch.cat(var_pred, dim=0)
    if tag == "train" or tag == "val":
        mean_gold = torch.cat(mean_gold, dim=0)
        var_gold = torch.cat(var_gold, dim=0)

    if DISCRETE_REG:
        mean_pred, var_pred = post_process(mean_pred, var_pred)

    if DISCRETE_CLS:
        mean_pred = un_discrete_label(mean_pred, "mean")
        var_pred = un_discrete_label(var_pred, "var")
        mean_gold = un_discrete_label(mean_gold, "mean", True)
        var_gold = un_discrete_label(var_gold, "var", True)

    if tag == "train" or tag == "val":
        _rmse_mean = rmse(mean_pred, mean_gold)
        _rmse_var = rmse(var_pred, var_gold)
        if tag == "train":
            return _rmse_mean, _rmse_var, _loss, _loss_mean, _loss_var
        else:
            return _rmse_mean, _rmse_var
    else:
        return mean_pred, var_pred
