import torch
import torch.nn as nn
from common import ResBlock
from common import default_conv



## Channel Weight (CW) Layer
class CWLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CWLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y

class MSCN_Block(nn.Module):
    def __init__(self, depth):
        super(MSCN_Block, self).__init__()
        channel = 64
        self.bottle = nn.Conv2d(in_channels = channel * depth, out_channels = channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_3_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_5_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv_5_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.CW = CWLayer(channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.bottle(x)
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.CW(self.relu(self.conv_5_2(input_2)))
        output = output_3_2 * output_5_2
        output = torch.add(output, identity_data)
        return output


class SRSpacesLearning(nn.Module):
    def __init__(self, feats, depth):
        super(SRSpacesLearning, self).__init__()
        self.conv = default_conv(feats, feats*4, 3, bias=True)
        self.up = nn.PixelShuffle(2)
        Mscnblocks = [MSCN_Block(i+1) for i in range(depth-1)]
        self.mscnblocks = nn.Sequential(*Mscnblocks)
        self.bottle = default_conv(depth, feats, 3, bias=True)
        self.rec = default_conv(feats, 3, 3, bias=True)
    
    def forward(self, x):
        x =self.conv(x)
        res = []
        out = self.up(x)
        res.append(out)
        for layer in self.mscnblocks:
            out = torch.cat(res,dim=1)
            out = layer(out)
            res.append(out)
        out = torch.cat(res,dim=1)
        out = self.bottle(out)
        out = self.rec(out)
        return out




class LRSpacesLearning(nn.Module):
    def __init__(self, feats, depth):
        super(LRSpacesLearning, self).__init__()
        resblocks = [ResBlock(default_conv, feats, 3) for i in range(depth-1)]
        self.resblocks = nn.Sequential(*resblocks)
        self.rec = default_conv(feats, 3, 3, bias=True)
    
    def forward(self, x):
        out1 = self.resblocks(x)
        out = self.rec(out1)
        return [out1, out]



  


class DoubleSV(nn.Module):
    def __init__(self, num_Channels, scale):
        super(DoubleSV, self).__init__()
        self.scale = scale
        self.featExt = nn.Sequential(
            nn.Conv2d(1,num_Channels,kernel_size=2*scale,stride=scale,padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_Channels//scale//scale,num_Channels,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )

        self.demosr = SRSpacesLearning(num_Channels,10)
        self.demo = LRSpacesLearning(num_Channels, 6)
        self.sr = SRSpacesLearning(num_Channels, 6)

    def forward(self, x):
        out = self.featExt(x)
        demosr = self.demosr(out)
        deepout, demos = self.demo(out)
        sr = self.sr(deepout)
        return demosr, sr, demos
        


