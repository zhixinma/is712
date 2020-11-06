import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import run_epoch
from const import *
from dataloader import read_img
from dataloader import ImgDataset

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_x = read_img(TEST_DIR, train=False)
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
best_model = torch.load(BEST_MODEL_PATH)

with torch.no_grad():
    test_pred_mean, test_pred_var = run_epoch(best_model, test_loader, "test")
    test_pred_mean = test_pred_mean.reshape(-1,).tolist()
    test_pred_var = test_pred_var.reshape(-1).tolist()

with open(RES_PATH, 'w') as f:
    for m, v in zip(test_pred_mean, test_pred_var):
        f.write(str(round(m, ROUND_NUM)) + '\t' + str(round(v, ROUND_NUM)) + "\n")
