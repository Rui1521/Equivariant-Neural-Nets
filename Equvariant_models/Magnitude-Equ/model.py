# Magnitude Equivariant Convolution Neural Nets
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################ResNet###########################
 
class scale_layer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation = True):
        super(scale_layer, self).__init__()
        self.input_channels = input_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.batchnorm = nn.BatchNorm2d(output_channels)
    
    def unfold(self, xx):
        out = F.pad(xx, ((self.pad_size, self.pad_size)*2), mode='replicate')
        out = F.unfold(out, kernel_size = self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, out.shape[-1])
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, xx.shape[-2], xx.shape[-1])
        return out
    
    def scale(self, xx):
        stds = xx.std((1,2,3), keepdim = True)
        out = xx/stds
        #out[out != out] = 0
        out = out.transpose(2,4).transpose(-1,-2)
        out = out.reshape(out.shape[0], self.input_channels, xx.shape[-2]*self.kernel_size, xx.shape[-1], self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, xx.shape[-2]*self.kernel_size, xx.shape[-1]*self.kernel_size)
        return out, stds.squeeze(2).squeeze(2)
    
    
    def scale_back(self, out, stds):
        out *= stds
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        return out
    
    def forward(self, xx):
        stds = xx.std()
        xx = xx/stds
        out = self.conv2d(xx)
        if self.activation:
            out = self.batchnorm(out)
            out = F.leaky_relu(out)
        out = self.scale_back(out, stds)
        return out
  
    
# 20-layer ResNet
class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, skip):
        super(Resblock, self).__init__()
        self.layer1 = scale_layer(input_channels, hidden_dim, kernel_size)
        self.layer2 = scale_layer(hidden_dim, hidden_dim, kernel_size)
        self.skip = skip
        
    def forward(self, x):
        out = self.layer1(x)
        if self.skip:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResNet, self).__init__()
        layers = [scale_layer(input_channels, 64, kernel_size)]
        layers += [Resblock(64, 64, kernel_size, True), Resblock(64, 64, kernel_size, True)]
        layers += [Resblock(64, 128, kernel_size, False), Resblock(128, 128, kernel_size, True)]
        layers += [Resblock(128, 256, kernel_size, False), Resblock(256, 256, kernel_size, True)]
        layers += [Resblock(256, 512, kernel_size, False), Resblock(512, 512, kernel_size, True)]
        layers += [scale_layer(512, output_channels, kernel_size, False)]
        self.model = nn.Sequential(*layers)
             
    def forward(self, xx):
        out = self.model(xx)
        return out