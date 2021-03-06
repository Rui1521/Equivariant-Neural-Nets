from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import time
from torch.autograd import Variable
from train import train_epoch, eval_epoch, test_epoch, Dataset, get_lr
from model import ResNet, U_net
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = "ResNet_2"

train_direc = "/global/cscratch1/sd/rwang2/Ocean/Data/Data/train/sample_"
valid_direc = "/global/cscratch1/sd/rwang2/Ocean/Data/Data/valid/sample_"
test_direc_time = "/global/cscratch1/sd/rwang2/Ocean/Data/Data/test/sample_"
test_direc_domain = "/global/cscratch1/sd/rwang2/Ocean/Data/Data/test_domain/sample_"
train_indices = list(range(7200)) 
valid_indices = list(range(1600))
test_indices_time = list(range(1600))
test_indices_domain = list(range(1600))

min_mse = 1
input_length = 30
learning_rate = 0.001
output_length = 3
batch_size = 64

train_set = Dataset(train_indices, input_length, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 40, output_length, valid_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)

model = nn.DataParallel(ResNet(input_channels = input_length*2, output_channels = 2, kernel_size = 3).to(device))
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
        torch.save(model, name + ".pth")
    end = time.time()
    if (len(train_mse) > 40 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break       
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"))


best_model = torch.load(name + ".pth")
loss_fun = torch.nn.MSELoss()
test_set = Dataset(test_indices_time, input_length, 40, 10, test_direc_time, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
rmse, preds, trues = eval_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "rmse": rmse}, 
            name +"_time.pt")


test_set = Dataset(test_indices_domain, input_length, 40, 10, test_direc_domain, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
rmse, preds, trues = eval_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "rmse": rmse}, 
            name +"_domain.pt")
