from . import common
import torch
import torch.nn as nn

def make_model(args):
    return Attemp1(args)

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

class MSCN_Block(nn.Module):
    def __init__(self):
        super(MSCN_Block, self).__init__()
        channel = 64
        self.conv_3_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_5_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.CW = CALayer(channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        output = self.CW(output_3_1 + output_5_1)
        output += x
        return output

class Attemp1(nn.Module):
    def __init__(self, args):
        super(Attemp1, self).__init__()
        scale = args.scale[0]
        out_channel = 64
        self.head =  nn.Sequential(*[nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.PReLU(),
                        nn.Conv2d(128, out_channel, kernel_size=3, padding=1),
                        nn.ReLU()]
                    )
        self.body = nn.Sequential(*[MSCN_Block() for i in range(10)])

        self.tail = nn.Sequential(*[nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                        nn.PixelShuffle(2), nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)])
    
    def forward(self, x):
        out = self.head(x)
        out1 = self.body(out)
        out1 += out
        out = self.tail(out1)

        return out
        
