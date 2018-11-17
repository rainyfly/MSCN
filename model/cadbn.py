import torch
import torch.nn as nn

def make_model(args, parent=False):
    return CADBPN()


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

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, padding=1, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##  Up-projection Block
def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, reduction, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_project = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)])
        self.rcan = RCAB(nn.Conv2d, nr, kernel_size=3, reduction=reduction, bias=True, bn=False, act=nn.ReLU(True) )
        

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        x = self.conv_project(x)
        out = self.rcan(x)

        return out

class CADBPN(nn.Module):
    def __init__(self):
        super(CADBPN, self).__init__()
        n0 = 256
        nr = 64
        scale = 2
        self.depth = 8
        self.color_extract =  nn.Sequential(*[
            nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(256//scale//scale,256,kernel_size=5,stride=1,padding=2),
            nn.PReLU(),
            nn.Conv2d(256, nr, 1),
            nn.PReLU(nr)
        ])

        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, nr, scale, 16, True, i > 1)
            )
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(
                DenseProjection(channels, nr, scale, 16, False, i != 0)
            )
            channels += nr
        
        self.reconstruction = nn.Sequential(*[nn.Conv2d(self.depth*nr,  64, kernel_size=3, padding=1), nn.ReLU(),
                             nn.Conv2d(64, 3, kernel_size=3, padding=1)]
                             )
    def forward(self, x):
        x = self.color_extract(x)
        h_list = []
        l_list = []
        for i in range(self.depth - 1):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        
        out = self.reconstruction(torch.cat(h_list, dim=1))
        return out



