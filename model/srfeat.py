import numpy as np
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(128,128,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(128,128,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.seq = nn.Sequential(self.conv1,self.bn1,self.act,self.conv2,self.bn2)
    def forward(self, x):
        out = self.seq(x)
        out += x
        return out

class SRFeat(nn.Module):
    def __init__(self):
        super(SRFeat, self).__init__()
        self.head = nn.Sequential(*[nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                        nn.PReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, padding=1),
                        nn.ReLU()]
                    )
        self.body = [ResidualBlock() for i in range(16)]
        self.bottles = [nn.Conv2d(128,128,kernel_size=1) for i in range(15)]
        self.tail = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.PixelShuffle(2),nn.Conv2d(64,3,kernel_size=3,padding=1))

    def forward(self, x):
        out = self.head(x)
        res = []
        for i in range(16):
            out = self.body[i](out)
            res.append(out)
        for i in range(15):
            out += self.bottles[i](res[i])
        
        out = self.tail(out)
        return out
        
        




