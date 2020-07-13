import torch
import math
import numpy as np
import torch.nn as nn
from torch.utils import data
import pytorch_lightning as pl
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv2d(pl.LightningModule):
    def __init__(self, out_channels, in_channels, kernel_size, l = 3, sout = 5, stride = 1, activation = True):
        super(Conv2d, self).__init__()
        self.out_channels= out_channels
        self.in_channels = in_channels
        self.l = l
        self.sout = sout
        self.activation = activation
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        weight_shape = (out_channels, l, 2, in_channels//2, kernel_size, kernel_size)
        self.stdv = math.sqrt(1. / (kernel_size * kernel_size * in_channels * l))
        self.weights = nn.Parameter(torch.Tensor(*weight_shape))
        self.reset_parameters()
        #self.batchnorm = nn.BatchNorm3d(sout)# affine=False
        self.stride = stride
        
    def reset_parameters(self):
        self.weights.data.uniform_(-self.stdv, self.stdv)
        if self.bias is not None:
            self.bias.data.fill_(0)
            
    def shrink_kernel(self, kernel, up_scale):
        up_scale = torch.tensor(up_scale).float()
        pad_in = (torch.ceil(up_scale**2).int())*((kernel.shape[2]-1)//2)
        pad_h = (torch.ceil(up_scale).int())*((kernel.shape[3]-1)//2)
        pad_w = (torch.ceil(up_scale).int())*((kernel.shape[4]-1)//2)
        padded_kernel = F.pad(kernel, (pad_w, pad_w, pad_h, pad_h, pad_in, pad_in))
        delta = up_scale%1
        if delta == 0:
            shrink_factor = 1
        else:
            # shrink_factor for coordinates if the kernel is over shrunk.
            shrink_factor = (((kernel.shape[4]-1))/(padded_kernel.shape[-1]-1)*(up_scale+1))
            # Adjustment to deal with weird filtering on the grid sample function.
            shrink_factor = 1.5*(shrink_factor-0.5)**3 + 0.57   

        grid = torch.meshgrid(torch.linspace(-1, 1, kernel.shape[2])*(shrink_factor**2),
                              torch.linspace(-1, 1, kernel.shape[3])*shrink_factor, 
                              torch.linspace(-1, 1, kernel.shape[4])*shrink_factor)

        grid = torch.cat([grid[2].unsqueeze(0).unsqueeze(-1), 
                          grid[1].unsqueeze(0).unsqueeze(-1), 
                          grid[0].unsqueeze(0).unsqueeze(-1)], dim = -1).repeat(kernel.shape[0],1,1,1,1)

        new_kernel = F.grid_sample(padded_kernel, grid.to(device))
        if kernel.shape[-1] - 2*up_scale > 0:
            new_kernel = new_kernel * (kernel.shape[-1]**2/((kernel.shape[-1] - 2*up_scale)**2 + 0.01))
        return new_kernel
    
    @staticmethod
    def dilate_kernel(kernel, dilation):
        if dilation == 0:
            return kernel 

        dilation = torch.tensor(dilation).float()
        delta = dilation%1

        d_in = torch.ceil(dilation**2).int()
        new_in = kernel.shape[2] + (kernel.shape[2]-1)*d_in

        d_h = torch.ceil(dilation).int()
        new_h = kernel.shape[3] + (kernel.shape[3]-1)*d_h

        d_w = torch.ceil(dilation).int()
        new_w = kernel.shape[4] + (kernel.shape[4]-1)*d_h

        new_kernel = torch.zeros(kernel.shape[0], kernel.shape[1], new_in, new_h, new_w)
        new_kernel[:,:,::(d_in+1),::(d_h+1), ::(d_w+1)] = kernel
        shrink_factor = 1
        # shrink coordinates if the kernel is over dilated.
        if delta != 0:
            new_kernel = F.pad(new_kernel, ((kernel.shape[4]-1)//2, (kernel.shape[4]-1)//2)*3)

            shrink_factor = (new_kernel.shape[-1] - 1 - (kernel.shape[4]-1)*(delta))/(new_kernel.shape[-1] - 1) 
            grid = torch.meshgrid(torch.linspace(-1, 1, new_in)*(shrink_factor**2), 
                                  torch.linspace(-1, 1, new_h)*shrink_factor, 
                                  torch.linspace(-1, 1, new_w)*shrink_factor)

            grid = torch.cat([grid[2].unsqueeze(0).unsqueeze(-1), 
                              grid[1].unsqueeze(0).unsqueeze(-1), 
                              grid[0].unsqueeze(0).unsqueeze(-1)], dim = -1).repeat(kernel.shape[0],1,1,1,1)

            new_kernel = F.grid_sample(new_kernel, grid)         
            #new_kernel = new_kernel/new_kernel.sum()*kernel.sum()
        return new_kernel[:,:,-kernel.shape[2]:]
    
    
    def forward(self, xx):
        #print(self.weights.shape, xx.shape)
        out = []
        for s in range(self.sout):
            t = np.minimum(s + self.l, self.sout)
            inp = xx[:,s:t].reshape(xx.shape[0], -1, xx.shape[-2], xx.shape[-1])
            w = self.weights[:,:(t-s),:,:,:].reshape(self.out_channels, 2*(t-s), self.in_channels//2, self.kernel_size, self.kernel_size).to(device)
            
            if (s - self.sout//2) < 0:
                new_kernel = self.shrink_kernel(w, (self.sout//2 - s)/2).to(device)
            elif (s - self.sout//2) == 1:
                new_kernel = torch.einsum('cvtxy,txysij->cvsij',w,dilationdict[w.shape[2]]).to(device)
                #new_kernel = self.dilate_kernel(w, (s - self.sout//2)/2).to(device)
            elif (s - self.sout//2) > 0:
                new_kernel = self.dilate_kernel(w, (s - self.sout//2)/2).to(device)
            else:
                new_kernel = w.to(device)
    
            new_kernel = new_kernel.reshape(self.out_channels, (t-s)*self.in_channels, new_kernel.shape[-2], new_kernel.shape[-1])
            conv = F.conv2d(inp, new_kernel, padding = ((new_kernel.shape[-2]-1)//2, (new_kernel.shape[-1]-1)//2))# bias = self.bias,
                 
            out.append(conv.unsqueeze(1))

        out = torch.cat(out, dim = 1) 
        #print(out.shape)
        if self.activation: 
            #out = self.batchnorm(out)
            out = F.leaky_relu(out)
        
        return out 
    
    
class Resblock(pl.LightningModule):
    def __init__(self, in_channels, hidden_dim, kernel_size, skip = True):
        super(Resblock, self).__init__()
        self.layer1 = Conv2d(out_channels = hidden_dim, in_channels = in_channels, kernel_size = kernel_size)
     
        self.layer2 = Conv2d(out_channels = hidden_dim, in_channels = hidden_dim, kernel_size = kernel_size) 
        
        self.skip = skip
        
    def forward(self, x):
        out = self.layer1(x)
        if self.skip:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out
    
class Scale_ResNet(pl.LightningModule):
    def __init__(self):
        super(Scale_ResNet, self).__init__()
        
        self.input_length = 25
        self.output_length = 3
        in_channels = self.input_length*2
        out_channels = 2 
        kernel_size = 5
        
        self.input_layer = Conv2d(out_channels = 32, in_channels = in_channels, kernel_size = kernel_size)        
        layers = [self.input_layer]
        layers += [Resblock(32, 32, kernel_size, True), Resblock(32, 32, kernel_size, True)]
        layers += [Resblock(32, 64, kernel_size, False), Resblock(64, 64, kernel_size, True)]
        layers += [Resblock(64, 128, kernel_size, False), Resblock(128, 128, kernel_size, True)]
        layers += [Resblock(128, 128, kernel_size, True), Resblock(128, 128, kernel_size, True)]
        layers += [Conv2d(out_channels = 2, in_channels = 128, kernel_size = kernel_size, sout = 1, activation = False)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, xx):
        out = self.model(xx)
        out = out.squeeze(1)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 0.001, betas=(0.9, 0.999), weight_decay=4e-4)
        return optimizer
    
    def loss_fun(self, preds, target):
        return torch.nn.MSELoss()(preds, target)  
    
    def blur_input(self, xx): 
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
    
    def setup(self, stage):
        direc = "/global/cscratch1/sd/rwang2/Equivariance/Ocean/Data/Ocean_Data_DeepCFD/Data/"
        train_direc = direc + "train/sample_"
        valid_direc = direc + "valid/sample_"
        test_direc = direc + "test/sample_"

        train_indices = list(range(7200)) 
        valid_indices = list(range(1600)) 
        test_indices = list(range(1600))   

        self.train_dataset = Dataset(train_indices, self.input_length, 40, self.output_length, train_direc)
        self.val_dataset = Dataset(valid_indices, self.input_length, 40, 6, valid_direc)
        self.test_dataset = Dataset(test_indices, self.input_length, 40, 10, test_direc) 
    
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size = 16, shuffle = True) 
    
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size = 16, shuffle = False)
    
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size = 16, shuffle = False)

    
    def training_step(self, train_batch, batch_idx):
        xx, yy = train_batch
        loss = 0
        ims = []
        train_rmse = []
        for i in range(yy.shape[2]):
            blur_xx = self.blur_input(xx)
            im = self.forward(blur_xx)
            xx = torch.cat([xx[:, :, 1:], im.unsqueeze(2)], 2)
            loss += self.loss_fun(im, yy[:,:,i])
        train_rmse = round(np.sqrt(loss.item()/yy.shape[2]), 5)
        return {'loss': loss, 'train_rmse': train_rmse}
    
    def validation_step(self, val_batch, batch_idx):
        xx, yy = val_batch
        loss = 0
        ims = []
        for i in range(yy.shape[2]):
            blur_xx = self.blur_input(xx)
            im = self.forward(blur_xx)
            xx = torch.cat([xx[:, :, 1:], im.unsqueeze(2)], 2)
            loss += self.loss_fun(im, yy[:,:,i])
            ims.append(im.unsqueeze(2))
                
        valid_rmse = round(np.sqrt(loss.item()/yy.shape[2]), 5)
        ims = torch.cat(ims, axis = 2)
        return {'val_loss': valid_rmse, 'preds': ims, "trues": yy}
    
    def validation_epoch_end(self, outputs):
        avg_loss = round(np.mean([x['val_loss'] for x in outputs]), 5)
        preds = torch.cat([x['preds'] for x in outputs], dim = 0).cpu().data.numpy()
        trues = torch.cat([x['trues'] for x in outputs], dim = 0).cpu().data.numpy()
        return {'valid_rmse': avg_loss, 'preds': preds, 'trues': trues}
    
    def test_step(self, test_batch, batch_idx):
        xx, yy = test_batch
        loss = 0
        ims = []
        for i in range(yy.shape[2]):
            blur_xx = self.blur_input(xx)
            im = self.forward(blur_xx)
            xx = torch.cat([xx[:, :, 1:], im.unsqueeze(2)], 2)
            loss += self.loss_fun(im, yy[:,:,i])
            ims.append(im.unsqueeze(2))
                
        test_rmse = round(np.sqrt(loss.item()/yy.shape[2]), 5)
        ims = torch.cat(ims, axis = 2)
        return {'test_loss': test_rmse, 'preds': ims, "trues": yy}
    
    def test_epoch_end(self, outputs):
        avg_loss = round(np.mean([x['test_loss'] for x in outputs]), 5)
        preds = torch.cat([x['preds'] for x in outputs], dim = 0).cpu().data.numpy()
        trues = torch.cat([x['trues'] for x in outputs], dim = 0).cpu().data.numpy()
        return {'test_rmse': avg_loss, 'preds': preds, 'trues': trues}
    
   

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
        x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].transpose(0,1)
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)].transpose(0,1)
        return x.float(), y.float()
    
    
class gaussain_blur(pl.LightningModule):
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
        xx = xx.reshape(xx.shape[0]*2, 1, xx.shape[2], xx.shape[3], xx.shape[4])
        xx = F.conv3d(xx, self.kernel, padding = (self.kernel.shape[-1]-1)//2)
        return xx.reshape(xx.shape[0]//2, 2, xx.shape[2], xx.shape[3], xx.shape[4])


def pre_compute_dilation(time_dims):
    dilationdict={}
    print("Generating Dilation mats")
    for time_dim in time_dims:
        print("Time_dim",time_dim)
        M = torch.Tensor(time_dim,5,5,time_dim,9,9)  #requires_grad = False 
        xx = torch.zeros(1,1,time_dim,5,5)
        for ti in tqdm.tqdm(range(time_dim)):
            for xi in range(5):
                for yi in range(5):
                    xx[0,0,ti,xi,yi] = 1.0
                    M[ti,xi,yi,:,:,:] = Conv2d.dilate_kernel(xx,0.5)
                    xx[0,0,ti,xi,yi] = 0.0
        dilationdict[time_dim] = M.to(device)
    return dilationdict

timedims = [16,25,32,64]
dilationdict = (pre_compute_dilation(timedims))

# class U_net(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(U_net, self).__init__()
#         self.conv1 = scale_conv2d(in_channels, 32, kernel_size = kernel_size, stride=2)
#         self.conv2 = scale_conv2d(32, 64, kernel_size = kernel_size, stride=2)
#         #self.conv2_2 = scale_conv2d(64, 64, kernel_size = kernel_size, stride = 1)
#         self.conv3 = scale_conv2d(64, 128, kernel_size = kernel_size, stride=2)
#         #self.conv3_1 = scale_conv2d(128, 128, kernel_size = kernel_size, stride=1)
#         self.conv4 = scale_conv2d(128, 256, kernel_size = kernel_size, stride=2)
#         #self.conv4_1 = scale_conv2d(512, 512, kernel_size = kernel_size, stride=1)

#         self.deconv3 = scale_deconv2d(256, 64)
#         self.deconv2 = scale_deconv2d(192, 32)
#         self.deconv1 = scale_deconv2d(96, 16)
#         self.deconv0 = scale_deconv2d(48, 8)
    
#         self.output_layer = scale_conv2d(8 + in_channels, out_channels, kernel_size=kernel_size, activation = False, sout = 1)

#     def forward(self, x):
#         #print(x.shape)
#         out_conv1 = self.conv1(x)
#         #print(out_conv1.shape)
#         out_conv2 = self.conv2(out_conv1)#)self.conv2_2(
#         #print(out_conv2.shape)
#         out_conv3 = self.conv3(out_conv2)#)self.conv3_1(
#         #print(out_conv3.shape)
#         out_conv4 = self.conv4(out_conv3)#)self.conv4_1(
#         #print(out_conv4.shape)
        
#         out_deconv3 = self.deconv3(out_conv4)
#         #print(out_conv3.shape, out_deconv3.shape)
#         concat3 = torch.cat((out_conv3, out_deconv3), 2)
#         out_deconv2 = self.deconv2(concat3)
#         #print(out_conv2.shape, out_deconv2.shape)
#         concat2 = torch.cat((out_conv2, out_deconv2), 2)
#         out_deconv1 = self.deconv1(concat2)
#         #print(out_conv1.shape, out_deconv1.shape)
#         concat1 = torch.cat((out_conv1, out_deconv1), 2)
#         out_deconv0 = self.deconv0(concat1)
#         #print(x.shape, out_deconv0.shape)
#         concat0 = torch.cat((x.reshape([x.shape[0], x.shape[1], -1, x.shape[4], x.shape[5]]), out_deconv0), 2)
#         out = self.output_layer(concat0)
#         out = out.squeeze(1)
#         return out