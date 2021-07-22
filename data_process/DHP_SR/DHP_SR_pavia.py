###########################################################################
# Created by: Yuxuan Zheng
# Email: yxzheng24@163.com
# Pytorch implementation of our DHP method for upsampling the LR-HSI, which is proposed in the paper titled "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network"

# Citation
# Y. Zheng, J. Li, Y. Li, J. Guo, X. Wu and J. Chanussot, "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network," 
# IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 11, pp. 8059-8076, Nov. 2020, doi: 10.1109/TGRS.2020.2986313.
###########################################################################

from __future__ import print_function

import numpy as np
import torch
import torch.optim
import scipy.io

from downsampler import Downsampler
from skip import skip
from utils import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


def sam(x_true,x_pre):

    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[ x, y,:], [-1])
            sa = np.reshape(x_true[x, y,:], [-1])
            tem1=np.dot(z,sa)
            tem2=(np.linalg.norm(z))*(np.linalg.norm(sa))
            samm[n]=np.arccos(tem1/tem2)
            n=n+1
    SAM=(np.mean(samm))*180/np.pi
    return SAM


file_name  = './pa_lhsconv/lhsconv_pa_b3.mat' # Here we take the image 'PaviaCenter_Patch11' as an example

# imgs contains ['HR'] ['LR'] ['bicubic'] ['nearest']
# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!
imgs = scipy.io.loadmat(file_name)

factor = 4 

print ('SAM bicubic: %.4f   SAM nearest: %.4f' %  (
                                    sam(imgs['HR'], imgs['bicubic']), 
                                    sam(imgs['HR'], imgs['nearest'])))


# # Set up parameters and net
input_depth = imgs['HR'].shape[2]

method =    '2D'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = 0.01
OPTIMIZER = 'adam'
num_iter = 10000
reg_noise_std = 0.01  # try 0 0.03 0.05 0.08

net_input = get_noise(input_depth, method, (imgs['HR'].shape[0], imgs['HR'].shape[1])).type(dtype).detach()

net = skip(input_depth, imgs['HR'].shape[2],
           num_channels_down = [128]*5,
           num_channels_up =   [128]*5,
           num_channels_skip =    [4]*5,  
           filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
           upsample_mode='bilinear',
           need1x1_up=False,
           need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# Loss function
# mse = torch.nn.MSELoss().type(dtype)
mae = torch.nn.L1Loss().type(dtype)

img_LR_var = imgs['LR'].transpose(2,0,1)
img_LR_var = torch.from_numpy(img_LR_var).type(dtype)
img_LR_var = img_LR_var[None, :].cuda()

downsampler = Downsampler(n_planes=imgs['HR'].shape[2], factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

# # Define closure and optimize
def closure():
    global i, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

#    total_loss = mse(out_LR, img_LR_var)
    total_loss = mae(out_LR, img_LR_var)
        
    total_loss.backward()

# Log
    sam_LR = sam(imgs['LR'].astype(np.float32), out_LR.detach().cpu().squeeze().numpy().transpose(1,2,0))
    sam_HR = sam(imgs['HR'].astype(np.float32), out_HR.detach().cpu().squeeze().numpy().transpose(1,2,0))
    print ('Iteration %05d    SAM_LR %.4f   SAM_HR %.4f' % (i, sam_LR, sam_HR), '\r', end='')

    sam_history.append([sam_HR])
    
    if i > 0 and sam_history[i] <= min(sam_history):
        out_HR_np = out_HR.detach().cpu().squeeze().numpy().transpose(1,2,0)
        scipy.io.savemat("./DHP_SR_results/mats_pavia/result_SRpab3.mat", {'pred':np.clip(out_HR_np, 0, 1)})
        f = open('./DHP_SR_results/txts_pavia/SAM_pab3.txt','w')
        f.write("Iter: " + str(i) + " SAM_HR: " + str(sam_HR)+'\n')
        f.close( )
    i += 1

    return total_loss

sam_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
