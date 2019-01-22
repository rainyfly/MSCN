import torch
import torch.nn as nn

from .common import  default_conv, BasicBlock, Upsampler

def make_model(args):
     return  MultiDCAN(64, 10, 2)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
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
        return x * y

class MultiDCABlock(nn.Module):
    def __init__(self, feats):
        super(MultiDCABlock, self).__init__()

        self.conv_3_1 = default_conv(feats, feats, 3)
        self.conv_3_2 = default_conv(feats*3, feats, 3)
        self.conv_5_1 = default_conv(feats, feats, 5)
        self.conv_5_2 = default_conv(feats*3, feats, 5)
        self.confusion = default_conv(feats*5, feats, 1)
        self.ca = CALayer(feats)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out_3_1 = self.relu(self.conv_3_1(x))
        out_5_1 = self.relu(self.conv_5_1(x))
        input_2 = torch.cat([x, out_3_1, out_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([x, out_3_1, out_5_1, output_3_2, output_5_2], 1)
        output = self.relu(self.confusion(input_3))
        output = self.ca(output)
        output = output + x

        return output

class MultiDCABGroup(nn.Module):
    def __init__(self, num_blocks, feats):
        super(MultiDCABGroup, self).__init__()

        self.blocks = nn.Sequential(*[MultiDCABlock(feats) for i in range(num_blocks)])
    
    def forward(self, x):
        out = self.blocks(x)
        out = out 
        return out


class MultiDCAN(nn.Module):
    def __init__(self, feats, num_group, scale):
        super(MultiDCAN, self).__init__()

        self.head = nn.Sequential(
            *[nn.Conv2d(1,feats*4,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.ReLU()
                        ]
        )
        self.body = nn.Sequential(*[MultiDCABGroup(3, 64) for i in range(num_group)])
        self.tail = nn.Sequential(*[Upsampler(default_conv, scale, feats), default_conv(feats, 3, 3)])
    
    def forward(self, x):
        out = self.head(x)
        out1 = self.body(out)
        out = out + out1
        out = self.tail(out)

        return out


                
        

        



