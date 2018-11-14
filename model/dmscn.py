from . import common
import torch
import torch.nn as nn

def make_model(args):
    return DMSRN(args)

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
        self.conv_3_1 = nn.Conv2d(in_channels = channel*depth, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_3_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_5_1 = nn.Conv2d(in_channels = channel*depth, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv_5_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.CW = CWLayer(channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.CW(self.relu(self.conv_5_2(input_2)))
        output = output_3_2 * output_5_2
        output = torch.add(output, identity_data)
        return output

class DMSRN(nn.Module):
    def __init__(self, args):
        super(DMSRN, self).__init__()
        scale = args.scale[0]
        out_channel = 64
        self.head =  nn.Sequential(*[nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                        nn.PReLU(),
                        nn.Conv2d(128, out_channel, kernel_size=3, padding=1),
                        nn.ReLU()]
                    )
        self.residual1 = self.make_layer(MSCN_Block, 1)
        self.residual2 = self.make_layer(MSCN_Block, 2)
        self.residual3 = self.make_layer(MSCN_Block, 3)
        self.residual4 = self.make_layer(MSCN_Block, 4)
        self.residual5 = self.make_layer(MSCN_Block, 5)
        self.residual6 = self.make_layer(MSCN_Block, 6)
        self.residual7 = self.make_layer(MSCN_Block, 7)
        self.residual8 = self.make_layer(MSCN_Block, 8)
        self.bottle = nn.Conv2d(in_channels= out_channel * 8 + 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.conv = nn.Conv2d(in_channels = 64, out_channels = 64 * scale * scale, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.convt = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias = True)


    def make_layer(self, block,i):
        layers = []
        layers.append(block(i))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = []
        out = self.head(x)
        LR = out
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual1(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual2(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual3(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual4(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual5(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual6(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual7(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.residual8(out)
        res.append(out)
        out = torch.cat(res,dim=1)
        out = self.bottle(out)
        out = self.convt(self.conv(out))
        out = self.conv_output(out)
        return out

