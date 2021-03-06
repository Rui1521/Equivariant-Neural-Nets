import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils import data
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.list_IDs = indices
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].squeeze(1)
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)].squeeze(1)
        return x.float(), y.float()
    
class gaussain_blur(nn.Module):
    def __init__(self, size, sigma, dim, channels):
        super(gaussain_blur, self).__init__()
        self.kernel = self.gaussian_kernel(size, sigma, dim, channels).to(device)
        
    def gaussian_kernel(self, size, sigma, dim, channels):

        kernel_size = 2*size + 1
        kernel_size = [kernel_size] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, channels, 1, 1, 1)

        return kernel
    
    def forward(self, xx):
        xx = xx.unsqueeze(1)
        xx = F.conv3d(xx, self.kernel, padding = (self.kernel.shape[-1]-1)//2)
        return xx.squeeze(1)
    
def blur_input(xx): 
    out = []
    for s in np.linspace(-1, 1, 5):
        if s > 0:
            blur = gaussain_blur(size = np.ceil(s), sigma = [s**2, s, s], dim  = 3, channels = 1).to(device)
            out.append(blur(xx).unsqueeze(1)*(s+1))
        elif s<0:
            out.append(xx.unsqueeze(1)*(1/(np.abs(s)+1)))
        else:
            out.append(xx.unsqueeze(1))
    out = torch.cat(out, dim = 1)
    return out


def train_epoch(train_loader, model, optimizer, loss_function):
    train_mse = []
    k = 0
    for xx, yy in train_loader:
        k += 1
        xx = xx.to(device)
        yy = yy.to(device)        
        loss = 0
        ims = []
        for i in range(yy.shape[1]):
            blur_xx = blur_input(xx)
            im = model(blur_xx)
            xx = torch.cat([xx[:, 1:], im], 1)
            loss += loss_function(im, yy[:,i].unsqueeze(1))        
        train_mse.append(loss.item()/yy.shape[1]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            for i in range(yy.shape[1]):
                blur_xx = blur_input(xx)
                im = model(blur_xx)
                xx = torch.cat([xx[:, 1:], im], 1)
                loss += loss_function(im, yy[:,i].unsqueeze(1))
                ims.append(im.cpu().data.numpy())
                
            valid_mse.append(loss.item()/yy.shape[1])    
            ims = np.concatenate(ims, axis = 1)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())

        try:
            preds = np.concatenate(preds, axis = 0)  
            trues = np.concatenate(trues, axis = 0)  
        except:
            print("can't concatenate")  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues