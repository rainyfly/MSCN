import torch.nn as nn
import torch
import numpy as numpy

class DSRN(nn.Module):
    def __init__(self,T):
        super(DSRN, self).__init__()
        self.T = T
        self.head = nn.Sequential(*[nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                        nn.PReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, padding=1),
                        nn.ReLU()]
                    )
        self.lrblocks = nn.ModuleList([LRblock() for i in range(T)])
        self.srblocks = nn.ModuleList([SRblock() for i in range(T)])
        self.resblocks = nn.ModuleList([nn.Conv2d(128,3,kernel_size=3,padding=1) for i in range(T)])

    def forward(self, x):
        out = self.head(x)
        sout = None
        for i in range(self.T):
            lout = self.lrblocks[i](out, sout, i)
            sout = self.srblocks[i](lout, sout, i)
            if i == 0:
                res = self.resblocks[i](sout)
            else:
                res += self.resblocks[i](sout)
        res = res/self.T
        return res




class LRblock(nn.Module):
    def __init__(self):
        super(LRblock, self).__init__()
        self.conv1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.seq = nn.Sequential(self.conv1,self.relu1,self.conv2,self.relu2)
        self.downsample = nn.Conv2d(128,128,kernel_size=3,stride=2, padding=1)
        self.prelu = nn.PReLU()
    
    def forward(self, lr, sr, i):
        residual = lr
        out = self.seq(lr)
        if i != 0:
            out1 = self.downsample(sr)
            out = out + residual + out1
        else:
            out = out + residual 
        out = self.prelu(out)
        return out

class SRblock(nn.Module):
    def __init__(self):
        super(SRblock, self).__init__()
        self.conv1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.seq = nn.Sequential(self.conv1,self.relu1,self.conv2,self.relu2)
        self.upsample = nn.ConvTranspose2d(128,128,kernel_size=4,stride=2, padding=1)
        self.prelu = nn.PReLU()
    def forward(self, lr, sr, i):
        out1 = self.upsample(lr)
        if i != 0:
            residual = sr
            out = self.seq(sr)
            out = out + residual + out1
        else:
            out = out1
        out = self.prelu(out)
        return out
