#Uniform Motion Equivariant Neural Nets
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
#########################UNET#######################################


class conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, um_dim = 2, activation = True, stride = 1, deconv = False, add_mean = True):
        super(conv2d, self).__init__()
        self.activation = activation
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.um_dim = um_dim 
        self.stride = stride
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.deconv = deconv
        
    def unfold(self, xx):
        if not self.deconv:
            xx = F.pad(xx, ((self.pad_size, self.pad_size)*2), mode='replicate')
        out = F.unfold(xx, kernel_size = self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, out.shape[-1])
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1])))
        if self.stride > 1:
            return out[:,:,:,:,::self.stride,::self.stride]
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
    
    def forward(self, xx):
        xx = self.unfold(xx)
        xx, avgs = self.subtract_mean(xx)
        out = self.conv2d(xx)
        if self.activation:
            #out = self.batchnorm(out)
            out = F.leaky_relu(out)
        if self.add_mean:
            out = self.add_mean(out, avgs)
        return out
    
class deconv2d(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(deconv2d, self).__init__()
        self.conv2d = conv2d(input_channels = input_channels, output_channels = output_channels, kernel_size = 4, um_dim = 2,
                             activation = True, stride = 1, deconv = True, add_mean = True)#
    
    def pad(self, xx):
        new_xx = torch.zeros(xx.shape[0], xx.shape[1], xx.shape[2]*2 + 3, xx.shape[3]*2 + 3)
        new_xx[:,:,:-3,:-3][:,:,::2,::2] = xx
        new_xx[:,:,1:-2,1:-2][:,:,::2,::2] = xx
        new_xx[:,:,-3:,-3:] = xx[:,:,-3:,-3:]
        return new_xx
    
    def forward(self, xx):
        out = self.pad(xx).to(device)
        return self.conv2d(out)

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv2d(input_channels, 64, kernel_size = kernel_size, stride=2)
        self.conv2 = conv2d(64, 128, kernel_size = kernel_size, stride=2)
        self.conv2_2 = conv2d(128, 128, kernel_size = kernel_size, stride = 1)
        self.conv3 = conv2d(128, 256, kernel_size = kernel_size, stride=2)
        self.conv3_1 = conv2d(256, 256, kernel_size = kernel_size, stride=1)
        self.conv4 = conv2d(256, 512, kernel_size = kernel_size, stride=2)
        self.conv4_1 = conv2d(512, 512, kernel_size = kernel_size, stride=1)

        self.deconv3 = deconv2d(512, 128)
        self.deconv2 = deconv2d(384, 64)
        self.deconv1 = deconv2d(192, 32)
        self.deconv0 = deconv2d(96, 16)
    
        self.output_layer = conv2d(16 + input_channels, output_channels, kernel_size=kernel_size, activation = False, add_mean = True)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_2(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        out_deconv3 = self.deconv3(out_conv4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)
        return out
 


############### ResNet ################
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
    
