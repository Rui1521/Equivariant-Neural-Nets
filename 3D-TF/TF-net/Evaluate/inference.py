#####################################################
# LOAD TRAINED MODEL AND MAKE PREDICTIONS
#####################################################

import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from turb_funcs import diagnostics_np
import os, sys


def minmaxscaler(data):
    """ scale large turbulence dataset by channel"""
    nsnaps = data.shape[0]
    dim = data.shape[1]
    nch = data.shape[4]

    # scale per channel
    data_scaled = []
    rescale_coeffs = []
    for i in range(nch):
        data_ch = data[:, :, :, :, i]
        minval = data_ch.min(axis=0)
        maxval = data_ch.max(axis=0)
        temp = (data_ch - minval) / (maxval - minval)
        data_scaled.append(temp)
        rescale_coeffs.append((minval, maxval))
    data_scaled = np.stack(data_scaled, axis=4)
    np.save('rescale_coeffs_3DHIT', rescale_coeffs)
    return data_scaled


def inverse_minmaxscaler(data, filename):
    """ Invert scaling using previously saved minmax coefficients """
    rescale_coeffs = np.load(filename)
    nsnaps = data.shape[0]
    dim = data.shape[1]
    nch = data.shape[4]

    # scale per channel
    data_orig = []
    for i in range(nch):
        data_ch = data[:, :, :, :, i]
        (minval, maxval) = rescale_coeffs[i]
        temp = data_ch * (maxval - minval) + minval
        data_orig.append(temp)
    data_orig = np.stack(data_orig, axis=4)
    return data_orig


def standardscaler(data):
    """ scale large turbulence dataset by channel"""
    nsnaps = data.shape[0]
    dim = data.shape[1]
    nch = data.shape[4]

    # scale per channel
    data_scaled = []
    rescale_coeffs = []
    for i in range(nch):
        data_ch = data[:, :, :, :, i]
        ch_mean = np.mean(data_ch)
        ch_std = np.std(data_ch)
        temp = (data_ch - ch_mean) / ch_std
        data_scaled.append(temp)
        rescale_coeffs.append((ch_mean, ch_std))
    data_scaled = np.stack(data_scaled, axis=4)
    np.save('rescale_coeffs_3DHIT', rescale_coeffs)
    return data_scaled


def convert_to_torchchannel(data):
    """ converts from  [snaps,dim1,dim2,dim3,nch] ndarray to [snaps,nch,dim1,dim2,dim3] torch tensor"""
    nsnaps = data.shape[0]
    dim1, dim2, dim3 = data.shape[1], data.shape[2], data.shape[3]
    nch = data.shape[-1]  # nch is last dimension in numpy input
    torch_permuted = np.zeros((nsnaps, nch, dim1, dim2, dim3))
    for i in range(nch):
        torch_permuted[:, i, :, :, :] = data[:, :, :, :, i]
    torch_permuted = th.from_numpy(torch_permuted)
    return torch_permuted


def convert_to_numpychannel_fromtorch(tensor):
    """ converts from [snaps,nch,dim1,dim2,dim3] torch tensor to [snaps,dim1,dim2,dim3,nch] ndarray """
    nsnaps = tensor.size(0)
    dim1, dim2, dim3 = tensor.size(2), tensor.size(3), tensor.size(4)
    nch = tensor.size(1)
    numpy_permuted = th.zeros(nsnaps, dim1, dim2, dim3, nch)
    for i in range(nch):
        numpy_permuted[:, :, :, :, i] = tensor[:, i, :, :, :]
    numpy_permuted = numpy_permuted.numpy()
    return numpy_permuted


def np_divergence(flow, grid):
    np_Udiv = np.gradient(flow[:, :, :, 0], grid[0])[0]
    np_Vdiv = np.gradient(flow[:, :, :, 0], grid[1])[1]
    np_Wdiv = np.gradient(flow[:, :, :, 0], grid[2])[2]
    np_div = np_Udiv + np_Vdiv + np_Wdiv
    total = np.sum(np_div) / (np.power(128, 3))
    return total


class CAEcurl(nn.Module):
    """
    Init and define stacked Conv Autoencoder layers and Physics layers 
    """

    def __init__(self, input_dim, input_size, batch, nfilters,
                 kernel_size, enc_nlayers, dec_nlayers):
        super(CAEcurl, self).__init__()
        self.il = input_dim[0]
        self.jl = input_dim[1]
        self.kl = input_dim[2]
        self.input_size = input_size  # no. of channels
        self.nfilters = nfilters
        self.batch = batch
        self.kernel_size = kernel_size
        self.output_size = 6  # 6 gradient components for 3 vector components of a CURL
        self.encoder_nlayers = enc_nlayers
        self.decoder_nlayers = dec_nlayers
        self.total_layers = self.encoder_nlayers + self.decoder_nlayers
        self.outlayer_padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2

        ############## Define Encoder layers
        encoder_cell_list = []
        for layer in range(self.encoder_nlayers):
            if layer == 0:
                cell_inpsize = self.input_size
                cell_outputsize = self.nfilters
                stridelen = 2
            else:
                cell_inpsize = self.nfilters
                cell_outputsize = self.nfilters
                stridelen = 2

            encoder_cell_list.append(nn.Conv3d(out_channels=cell_outputsize, in_channels=cell_inpsize,
                                               kernel_size=self.kernel_size, stride=stridelen).cuda())
            # accumulate layers
        self.encoder_cell_list = th.nn.ModuleList(encoder_cell_list)

        ############## Define Decoder layers
        decoder_cell_list = []
        for layer in range(self.decoder_nlayers):
            if layer == (self.decoder_nlayers - 1):
                cell_inpsize = self.nfilters
                cell_outputsize = self.input_size
                cell_padding = self.outlayer_padding
                stridelen = 2
            else:
                cell_inpsize = self.nfilters
                cell_outputsize = self.nfilters
                cell_padding = 0
                stridelen = 2

            decoder_cell_list.append(nn.ConvTranspose3d(out_channels=cell_outputsize, in_channels=cell_inpsize,
                                                        kernel_size=self.kernel_size, stride=stridelen,
                                                        output_padding=cell_padding).cuda())
            # accumulate layers
        self.decoder_cell_list = th.nn.ModuleList(decoder_cell_list)

        ############# Physics Layers
        # d/dx
        self.ddxKernel = th.zeros(3, 3, 3)
        self.ddxKernel[1, 0, 1] = -0.5
        self.ddxKernel[1, 2, 1] = 0.5
        # d/dy
        self.ddyKernel = th.zeros(3, 3, 3)
        self.ddyKernel[0, 1, 1] = -0.5
        self.ddyKernel[2, 1, 1] = 0.5
        # d/dz
        self.ddzKernel = th.zeros(3, 3, 3)
        self.ddzKernel[1, 1, 0] = -0.5
        self.ddzKernel[1, 1, 2] = 0.5
        #### declare weights
        self.weights = th.zeros((self.output_size, self.input_size, 3, 3, 3))
        self.weights = self.weights.type(th.cuda.FloatTensor)
        # dfy/dx
        self.weights[0, 0, ::] = th.zeros(3, 3, 3)
        self.weights[0, 1, ::] = self.ddxKernel.clone()
        self.weights[0, 2, ::] = th.zeros(3, 3, 3)
        # dfz/dx
        self.weights[1, 0, ::] = th.zeros(3, 3, 3)
        self.weights[1, 1, ::] = th.zeros(3, 3, 3)
        self.weights[1, 2, ::] = self.ddxKernel.clone()
        # dfx_dy
        self.weights[2, 0, ::] = self.ddyKernel.clone()
        self.weights[2, 1, ::] = th.zeros(3, 3, 3)
        self.weights[2, 2, ::] = th.zeros(3, 3, 3)
        # dfz_dy
        self.weights[3, 0, ::] = th.zeros(3, 3, 3)
        self.weights[3, 1, ::] = th.zeros(3, 3, 3)
        self.weights[3, 2, ::] = self.ddyKernel.clone()
        # dfx_dz
        self.weights[4, 0, ::] = self.ddzKernel.clone()
        self.weights[4, 1, ::] = th.zeros(3, 3, 3)
        self.weights[4, 2, ::] = th.zeros(3, 3, 3)
        # dfy_dz
        self.weights[5, 0, ::] = th.zeros(3, 3, 3)
        self.weights[5, 1, ::] = self.ddzKernel.clone()
        self.weights[5, 2, ::] = th.zeros(3, 3, 3)
        ### Boundary padding
        self.correctBoundaryTensor = th.ones([self.batch, self.output_size, self.il, self.jl, self.kl])
        # du/dx BC correction for one-sided difference at boundaries with padding
        self.correctBoundaryTensor[:, 0, :, -1, :] = 2.0
        self.correctBoundaryTensor[:, 0, :, 0, :] = 2.0
        self.correctBoundaryTensor[:, 1, :, -1, :] = 2.0
        self.correctBoundaryTensor[:, 1, :, 0, :] = 2.0
        # du/dy BC correction for one-sided difference at boundaries with padding
        self.correctBoundaryTensor[:, 2, 0, :, :] = 2.0
        self.correctBoundaryTensor[:, 2, -1, :, :] = 2.0
        self.correctBoundaryTensor[:, 3, 0, :, :] = 2.0
        self.correctBoundaryTensor[:, 3, -1, :, :] = 2.0
        # du/dz BC correction for one-sided difference at boundaries with padding
        self.correctBoundaryTensor[:, 4, :, :, 0] = 2.0
        self.correctBoundaryTensor[:, 4, :, :, -1] = 2.0
        self.correctBoundaryTensor[:, 5, :, :, 0] = 2.0
        self.correctBoundaryTensor[:, 5, :, :, -1] = 2.0
        self.correctBoundaryTensor = self.correctBoundaryTensor.type(th.cuda.FloatTensor)
        self.register_buffer('r_correctBoundaryTensor', self.correctBoundaryTensor)
        ### define curl operation
        self.rep_pad = nn.ReplicationPad3d(1)
        self.curlConv = nn.Conv3d(self.input_size, self.output_size, 3, bias=False, padding=0)
        with th.no_grad():
            self.curlConv.weight = nn.Parameter(self.weights)

    def forward(self, x):
        """ Forward Pass """
        # distbatchsize = x.size(0)
        cur_input = x.cuda()

        # Encoder
        for layer in range(self.encoder_nlayers):
            x = th.relu(self.encoder_cell_list[layer](x))
            # print('x enc size:', x.size())
        # Decoder
        for layer in range(self.decoder_nlayers):
            x = th.relu(self.decoder_cell_list[layer](x))
            # print('x dec size:', x.size())

        # Physics Layers
        x = self.rep_pad(x)
        output = self.curlConv(x)  # compute conv

        curlGrad = th.zeros([self.batch, self.output_size, self.il, self.jl, self.kl])
        curlGrad = curlGrad.type(th.cuda.FloatTensor)
        curlField = th.zeros([self.batch, self.input_size, self.il, self.jl, self.kl])
        curlField = curlField.type(th.cuda.FloatTensor)
        for i in range(self.output_size):
            curlGrad[:, i, ::] = output[:, i, ::] * self.r_correctBoundaryTensor[:, i, ::]
        # construct curl vector
        curlField[:, 0, ::] = curlGrad[:, 3, ::] - curlGrad[:, 5, ::]
        curlField[:, 1, ::] = curlGrad[:, 4, ::] - curlGrad[:, 1, ::]
        curlField[:, 2, ::] = curlGrad[:, 0, ::] - curlGrad[:, 2, ::]

        return curlField


path = '/home/arvindm/datasets/ScalarHIT/128cube/scalarHIT_fields100.h5'
f = h5py.File(path, 'r')
fields = f['fields']
nch = 3
input_dim = (128, 128, 128)
nfilters = 15
nsnaps = 20
data = fields[:nsnaps, :, :, :, :nch]  # all 5 fields including passive scalars
data = minmaxscaler(data)  # scale data 0 - 1
epochs = 1000000
batch_size = 4
kernel_size = (3, 3, 3)
enc_nlayers = 3
dec_nlayers = 3
nGPU = th.cuda.device_count()
distbatch_size = int(batch_size / nGPU)
nbatches = nsnaps - (nsnaps % batch_size)
data = data[:nbatches, ::]

# reshape data for pytorch input and conver to torch tensor
inp_tensor = convert_to_torchchannel(data)
print(inp_tensor.size())

# Create dataloader pipeline
train_dataset = th.utils.data.TensorDataset(inp_tensor, inp_tensor)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# define model
model = CAEcurl(input_dim, nch, distbatch_size, nfilters,
                kernel_size, enc_nlayers, dec_nlayers)
"""
if th.cuda.device_count() > 1:
    print('using',  th.cuda.device_count(), 'GPUs')
    model = nn.DataParallel(model) 
"""
# model.cuda()

criterion = nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=1e-04,
                          weight_decay=1e-5)

# Encode data
print('Loading checkpoint to GPU...')
modelpath = '/home/arvindm/MELT/DFD2019/r1/divFree/v1/checkpoints'
checkpoint = th.load(modelpath, map_location="cuda:0")
# checkpoint = th.load(modelpath)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
# print('loading model %s for evaluation, trained for %5d epochs with %8.8f final loss' % ( str(blkID), epoch, loss))
print('loading model')
model.cuda()
model.eval()

# save 2 diagnostics datasets

# 1
randidx = np.random.randint(0, 10)
print('rand idx is', randidx)
test_input = inp_tensor[randidx:randidx + distbatch_size, ::].cuda()
print(test_input.size())
real_space = model(test_input.type(th.cuda.FloatTensor))
print(real_space.size())

ytest = convert_to_numpychannel_fromtorch(test_input.detach())

predt = convert_to_numpychannel_fromtorch(real_space.detach())

# rescale to physical limits
dns = inverse_minmaxscaler(ytest, 'rescale_coeffs_3DHIT.npy')
mod = inverse_minmaxscaler(predt, 'rescale_coeffs_3DHIT.npy')

dns = dns[0, :, :, :, :3]
mod = mod[0, :, :, :, :3]
print(dns.shape)

# Calculate divergence
dx = (2 * np.pi) / input_dim[0]
dy = (2 * np.pi) / input_dim[0]
dz = (2 * np.pi) / input_dim[0]
dns_div = np_divergence(dns, [dx, dy, dz])
print('DNS divergence is', dns_div)
model_div = np_divergence(mod, [dx, dy, dz])
print('Model divergence is', model_div)

# diagnostics_np(mod,dns,save_dir='/home/arvindm/MELT/DFD2019/r1/divFree/v1/diagnosticsCAE/', iteration=1, pos=[0,0,1], dx=[0.049, 0.049, 0.049], diagnostics=['spectrum', 'intermittency', 'structure_functions','QR'])
