import torch
import torch.nn as nn

from common import ResBlock
from common import default_conv

def make_model(args):
    return HighSpace(128, 2)

# 15 resblock in high-feature space
class HighSpace(nn.Module):
    def __init__(self, num_Channels, scale):
        super(HighSpace, self).__init__()
        self.scale = scale
        self.featExt = nn.Sequential(
            nn.Conv2d(1,num_Channels,kernel_size=2*scale,stride=scale,padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_Channels//scale//scale,num_Channels,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )
        self.up = nn.PixelShuffle(2)
        self.Conv = default_conv(num_Channels//scale//scale, num_Channels, 3)
        self.res = nn.Sequential(*[ResBlock(default_conv, num_Channels, 3) for i in range(15)])
        self.rec = default_conv(num_Channels, 3, 3)

    def forward(self, x):
        out = self.featExt(x)
        out = self.up(out)
        out = self.Conv(out)
        out = self.res(out)
        out = self.rec(out)
        return out
    


