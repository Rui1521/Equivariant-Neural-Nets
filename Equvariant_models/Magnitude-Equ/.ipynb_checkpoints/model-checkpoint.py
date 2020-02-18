# Magnitude Equivariant Convolution Neural Nets
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################Unet##################################################
class conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation = True, stride = 1, deconv = False):
        super(conv2d, self).__init__()
        self.input_channels = input_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size, bias = False)
        self.pad_size = (kernel_size - 1)//2
        #if not deconv:
         #   self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride =  stride, padding = self.pad_size, bias = True)
        #else:
         #   self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, bias = True)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.leakyrelu = nn.LeakyReLU()
        self.deconv = deconv
        self.stride = stride
        
    def unfold(self, xx):
        if not self.deconv:
            out = F.pad(xx, ((self.pad_size, self.pad_size)*2), mode='replicate')#
        else:
            out = xx
        out = F.unfold(out, kernel_size = self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, out.shape[-1])
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1])))
        if self.stride > 1:
            return out[:,:,:,:,::self.stride,::self.stride]
        return out
    
    def scale(self, xx):
        #norm  = torch.sqrt(xx[:,::2]**2 + xx[:,1::2]**2).mean((1,2,3), keepdim = True).unsqueeze(2)
        #out = xx.reshape(xx.shape[0], xx.shape[1]//2, 2, xx.shape[2], xx.shape[3], xx.shape[4], xx.shape[5])/norm        
        #norm  = norm.squeeze(1)
        #out = out.reshape(out.shape[0], out.shape[1]*2, out.shape[3], out.shape[4], out.shape[5], out.shape[6])
        stds = xx.std((1,2,3), keepdim = True)
        avgs = xx.mean((1,2,3), keepdim = True)
        #out = xx/(stds + 10e-7)
        out = (xx-avgs) / (stds + 10e-7)
        out = out.transpose(2,4).transpose(-1,-2)
        out = out.reshape(out.shape[0], self.input_channels, xx.shape[-2]*self.kernel_size, xx.shape[-1], self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, xx.shape[-2]*self.kernel_size, xx.shape[-1]*self.kernel_size)
        return out, avgs.squeeze(2).squeeze(2), stds.squeeze(2).squeeze(2) #
    
    def scale_back(self, out, avgs, stds):#avgs, 
        out = out*(stds + 10e-7) + avgs
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        return out
    
    def forward(self, xx):
        out = self.unfold(xx)
        out, avgs, stds = self.scale(out)#
        #stds = xx.std
        #norm = torch.sqrt(xx[:,::2]**2 + xx[:,1::2]**2).mean((1,2,3), keepdim  = True)
        #out  = xx/(norm +  1e-5)
        out = self.conv2d(out)
        #out[out!=out] = torch.mean(out)
        if self.activation:
            #out = self.batchnorm(out)
            out = self.leakyrelu(out)
        out = self.scale_back(out, avgs,stds)
        return out
    
    
class deconv2d(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(deconv2d, self).__init__()
        self.conv2d = conv2d(input_channels = input_channels, output_channels = output_channels, kernel_size = 4, activation = True, stride = 1, deconv = True)
    
    def pad(self, xx):
        new_xx = torch.zeros(xx.shape[0], xx.shape[1], xx.shape[2]*2 + 3, xx.shape[3]*2 + 3)
        new_xx[:,:,:-3,:-3][:,:,::2,::2] = xx
        new_xx[:,:,:-3,:-3][:,:,1::2,1::2] = xx
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
    
        self.output_layer = conv2d(16 + input_channels, output_channels, kernel_size=kernel_size, activation = False)

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
            #out = self.batchnorm(out)
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