import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

def make_model(args, parent=False):
    return IDN()
def Inter_Bicubic(x, scale):
    x_numpy = x.data.cpu().numpy()
    x_resize = np.random.random([x_numpy.shape[0],x_numpy.shape[1],x_numpy.shape[2]*scale,x_numpy.shape[3]*scale])

    for i in range(x_numpy.shape[0]):

        x_resize[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        x_resize[i,1,:,:] = cv2.resize(x_numpy[i,1,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        x_resize[i,2,:,:] = cv2.resize(x_numpy[i,2,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)

    return  Variable(torch.from_numpy(x_resize).float().cuda(), volatile=False)

# DBlocks
class Enhancement_unit(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(Enhancement_unit, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))       
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-2*nDiff, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-2*nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.Sequential(*block_0)

        block_1 = [] 
        block_1.append(nn.Conv2d(nFeat-nFeat//4, nFeat, kernel_size=3, padding=1, bias=True))        
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat+nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.Sequential(*block_1)
        self.compress = nn.Conv2d(nFeat+nDiff, nFeat, kernel_size=1, padding=0, bias=True)
    def forward(self, x):

        x_feature_shot = self.conv_block0(x)
        feature = x_feature_shot[:,0:(self.D3-self.D3//self.s),:,:]
        feature_slice = x_feature_shot[:,(self.D3-self.D3//self.s):self.D3,:,:]
        x_feat_long = self.conv_block1(feature)
        feature_concat = torch.cat((feature_slice, x), 1)
        out = x_feat_long + feature_concat
        out = self.compress(out)
        return out


class IDN(nn.Module):
    def __init__(self):
        super(IDN, self).__init__()
        nFeat = 64
        nDiff = 16
        nFeat_slice = 4
        nChannel = 3
        self.scale = 2
        self.entry = nn.Sequential(*[nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                        nn.PReLU(),
                        nn.Conv2d(128, nFeat, kernel_size=3, padding=1),
                        nn.ReLU()]
                    )


        self.Enhan_unit1 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit2 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit3 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit4 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        # Upsampler
        #self.upsample = nn.ConvTranspose2d(nFeat, nChannel, stride=3, kernel_size=17, padding=8)
        self.tail = nn.Sequential(nn.Conv2d(nFeat*5,nFeat*4,kernel_size=1),nn.PixelShuffle(2),nn.Conv2d(nFeat,nChannel,kernel_size=3,padding=1))


    def forward(self, x):
        
        x = self.entry(x)
        uouts = []
        uouts.append(x)
        x = self.Enhan_unit1(x)
        uouts.append(x)
        x = self.Enhan_unit2(x)
        uouts.append(x)
        x = self.Enhan_unit3(x)
        uouts.append(x)
        x = self.Enhan_unit4(x)
        uouts.append(x)
        x = torch.cat(uouts, dim = 1)

        out = self.tail(x)

        return out  





