#Uniform Motion Equivariant Neural Nets
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ ResNet ###################

class mean_layer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, um_dim = 2, activation = True, stride = 1):
        super(mean_layer, self).__init__()
        self.activation = activation
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.um_dim = um_dim 
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size*stride, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.batchnorm = nn.BatchNorm2d(output_channels)
    
    def unfold(self, xx):
        out = F.pad(xx, ((self.pad_size, self.pad_size)*2), mode='replicate')
        out = F.unfold(out, kernel_size = self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, out.shape[-1])
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1])
        return out
    
    def subtract_mean(self, xx):
        out = xx.reshape(xx.shape[0], self.input_channels//self.um_dim, self.um_dim, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1])
        avgs = out.mean((1,3,4), keepdim=True)
        out -= avgs
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1]).transpose(2,4).transpose(-1,-2)
        out = out.reshape(out.shape[0], self.input_channels, xx.shape[-2]*self.kernel_size, xx.shape[-1], self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, xx.shape[-2]*self.kernel_size, xx.shape[-1]*self.kernel_size)
        return out, avgs.squeeze(3).squeeze(3)
    
    
    def add_mean(self, out, avgs):
        out = out.reshape(out.shape[0], out.shape[1]//self.um_dim, self.um_dim, out.shape[-2], out.shape[-1])
        out += avgs
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        return out
    
    def forward(self, xx, add_mean = True):
        xx = self.unfold(xx)
        xx, avgs = self.subtract_mean(xx)
        out = self.conv2d(xx)
        if self.activation:
            out = self.batchnorm(out)
            out = F.leaky_relu(out)
        if add_mean:
            out = self.add_mean(out, avgs)
        return out



# 20-layer ResNet
class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, skip):
        super(Resblock, self).__init__()
        self.layer1 = mean_layer(input_channels, hidden_dim, kernel_size)
        self.layer2 = mean_layer(hidden_dim, hidden_dim, kernel_size)
        self.skip = skip
        
    def forward(self, x):
        out = self.layer1(x)
        if self.skip:
            out = self.layer2(out, False) + x
        else:
            out = self.layer2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResNet, self).__init__()
        layers = [mean_layer(input_channels, 64, kernel_size)]
        layers += [Resblock(64, 64, kernel_size, False), Resblock(64, 64, kernel_size, False)]
        layers += [Resblock(64, 128, kernel_size, False), Resblock(128, 128, kernel_size, False)]
        layers += [Resblock(128, 256, kernel_size, False), Resblock(256, 256, kernel_size, False)]
        layers += [Resblock(256, 512, kernel_size, False), Resblock(512, 512, kernel_size, False)]
        layers += [mean_layer(512, output_channels, kernel_size, activation = False)]
        self.model = nn.Sequential(*layers)
             
    def forward(self, xx):
        out = self.model(xx)
        return out
    

