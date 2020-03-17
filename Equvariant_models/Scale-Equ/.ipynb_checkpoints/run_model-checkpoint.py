import os
import time
import torch
import torch.nn as nn     
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.stats import norm
from torch.utils import data
from model2d_scalar import ResNet, scale_cnn2d
from train_scalar import train_epoch, eval_epoch, Dataset, get_lr, blur_input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_direc = ""
idx = str(1)
learning_rate = 0.001
min_mse = 1
output_length = 3
batch_size = 16
input_length = 
train_indices = ""
valid_indices = ""
test_indices = ""

train_set = Dataset(train_indices, input_length, 40, output_length, train_direc)
valid_set = Dataset(valid_indices, input_length, 40, 6, train_direc)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)


print("Initializing...")
model = nn.DataParallel(scale_cnn2d(out_channels = 1, in_channels = input_length, kernel_size = 5, hidden_dim = 64, num_layers = 8).to(device))
print("Done")

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()


train_mse = []
valid_mse = []
test_mse = []

for i in range(100):
    start = time.time()
    scheduler.step()

    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(model, "CNN-Scale"+idx+".pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), idx)


test_direc = "
loss_fun = torch.nn.MSELoss()
test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
mse, preds, trues = eval_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "loss_curve": mse}, 
            "CNN-Scale"+idx+".pt")
