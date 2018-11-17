from model import common

import torch.nn as nn

def make_model(args, parent=False):
    args.n_resblocks = 16
    args.n_feats = 256
    return EDSR(args)

## EDSR
class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
  
        kernel_size = 3

        scale = args.scale[0]
        act = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1)
        self.subpix1 = nn.PixelShuffle(scale)
        self.conv2 = nn.Conv2d(256//scale//scale,256,kernel_size=5,stride=1,padding=2)
        self.prelu1 = nn.PReLU()
        # RGB mean for DIV2K
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(256, n_feats, kernel_size)]

        # define body module
        modules_body = [common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)]

            

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        #self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.subpix1(x)
        x = self.conv2(x)
        x = self.prelu1(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        #x = self.add_mean(x)

        return x 

