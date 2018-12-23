import torch
import torch.nn as nn

def make_model(args):
    return ResNetDemSR(24,256,2)
    
class BasicBlock(nn.Module):
  def __init__(self,in_channels,out_channels,stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
    self.prelu1 = nn.PReLU()
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
    self.prelu2 = nn.PReLU()
  def forward(self,x):
    out = self.conv1(x)
    out = self.prelu1(out)
    out = self.conv2(out)
    out += x
    out = self.prelu2(out)
    return out


class ResNetDemSR(nn.Module):
  def __init__(self,num_Resblocks,num_Channels,scale):
    super(ResNetDemSR,self).__init__()
    self.num_Resblocks = num_Resblocks
    self.num_Channels = num_Channels
    self.scale = scale
    self.conv1 = nn.Conv2d(1,num_Channels,kernel_size=2*scale,stride=scale,padding=1)
    self.subpix1 = nn.PixelShuffle(scale)
    self.conv2 = nn.Conv2d(num_Channels//scale//scale,num_Channels,kernel_size=5,stride=1,padding=2)
    self.prelu1 = nn.PReLU()
    layers = []
    for i in range(num_Resblocks):
      layers.append(BasicBlock(num_Channels,num_Channels))
    self.res = nn.Sequential(*layers)
    # self.subpix2 = SubPixelConv(scale)
    self.subpix2 = nn.PixelShuffle(scale)
    self.conv3 = nn.Conv2d(num_Channels//scale//scale,num_Channels,kernel_size=5,stride=1,padding=2)
    self.prelu2 = nn.PReLU()
    self.conv4 = nn.Conv2d(num_Channels,3,kernel_size=3,stride=1,padding=1)
  
  def forward(self,x):
    out = self.conv1(x)  
    out = self.subpix1(out)
    out = self.conv2(out)
    out = self.prelu1(out)
    out = self.res(out)
    out = self.subpix2(out)
    out = self.conv3(out)
    out = self.prelu2(out)
    out = self.conv4(out) 

    return out
