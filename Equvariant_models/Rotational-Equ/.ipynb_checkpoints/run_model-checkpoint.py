import os
import matplotlib.pyplot as plt
import torch
import dill
from e2cnn import gspaces
from e2cnn import nn
import numpy as np
import time
from torch.utils import data
import warnings
from model import ResNet, U_net
from train import train_epoch, eval_epoch, test_epoch, Dataset, get_lr
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = str(1)

output_length = 3
input_length = 24
min_mse = 1
learning_rate = 0.001
batch_size = 16
train_direc = ""

train_indices = 
valid_indices = 

train_set = Dataset(train_indices, input_length, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 40, 6, train_direc, True) # 6
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)

print("Initializing")
model = torch.nn.DataParallel(U_net(input_frames = input_length, output_frames = 1, kernel_size = 3, N = 8).to(device))
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
    mse, _, _ = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(model.state_dict(), "Unet-rot"+idx+".pth")

    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(i, train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), idx)
print("***************")

test_indices = 
#best_model = torch.nn.DataParallel(U_net(input_frames = input_length, output_frames = 1, kernel_size = 3, N = 8).to(device))
#best_model.load_state_dict(torch.load("Unet-rot"+idx+".pth"))
loss_fun = torch.nn.MSELoss()
test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
valid_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "loss_curve": loss_curve}, 
            "Unet-rot"+idx+".pt")

