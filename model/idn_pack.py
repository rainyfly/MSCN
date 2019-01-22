import torch
import torch.nn as nn
from torch.autograd import Function

def make_model(arg):
    return IDNPack()

class PackBayerMosaicLayer(nn.Module):
    def __init__(self):
        super(PackBayerMosaicLayer, self).__init__()

    def forward(self, x):
        return packbayermosaiclayer.apply(x)
    
class packbayermosaiclayer(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        n, c, h, w = x.size()
        xx = x.clone()
        # out = nn.Parameter(torch.Tensor(n, 4, h//2, w//2), requires_grad=True)
        out = torch.zeros((n, 4, h//2, w//2), dtype=xx.dtype, device='cuda')
        out[:, 0, :, :] = xx[:, 1, ::2, ::2] # G
        out[:, 1, :, :] = xx[:, 0, ::2, 1::2] # R
        out[:, 2, :, :] = xx[:, 2, 1::2, ::2] # B
        out[:, 3, :, :] = xx[:, 1, 1::2, 1::2] # G
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x,  = ctx.saved_tensors
        n, c, h, w = x.size()
        # print("PackBayerMosaicLayer backward recalled !")
        # grad_in = nn.Parameter(torch.Tensor(n, c, h, w), requires_grad=True)
        grad_in = torch.zeros_like(x, dtype=x.dtype, requires_grad=True)
        # grad_in = x.clone()
        grad_in[:, 1, ::2, ::2]   = grad_output[:, 0, :, :]
        grad_in[:, 0, ::2, 1::2]   = grad_output[:, 1, :, :]
        grad_in[:, 2, 1::2, ::2]   = grad_output[:, 2, :, :]
        grad_in[:, 1, 1::2, 1::2]  = grad_output[:, 3, :, :]
        return grad_in

# DBlocks
class Enhancement_unit(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(Enhancement_unit, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))       
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-2*nDiff, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-2*nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.Sequential(*block_0)

        block_1 = [] 
        block_1.append(nn.Conv2d(nFeat-nFeat//4, nFeat, kernel_size=3, padding=1, bias=True))        
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat+nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.Sequential(*block_1)
        self.compress = nn.Conv2d(nFeat+nDiff, nFeat, kernel_size=1, padding=0, bias=True)
    def forward(self, x):

        x_feature_shot = self.conv_block0(x)
        feature = x_feature_shot[:,0:(self.D3-self.D3//self.s),:,:]
        feature_slice = x_feature_shot[:,(self.D3-self.D3//self.s):self.D3,:,:]
        x_feat_long = self.conv_block1(feature)
        feature_concat = torch.cat((feature_slice, x), 1)
        out = x_feat_long + feature_concat
        out = self.compress(out)
        return out


class IDNPack(nn.Module):
    def __init__(self):
        super(IDNPack, self).__init__()
        nFeat = 64
        nDiff = 16
        nFeat_slice = 4
        nChannel = 3
        self.scale = 2
        self.pack = PackBayerMosaicLayer()
        self.entry = nn.Sequential(*[
                        nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1),
                        nn.PReLU(),
                        nn.Conv2d(128, nFeat*4, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.PixelShuffle(2)]
                    )


        self.Enhan_unit1 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit2 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit3 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit4 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        # Upsampler
        #self.upsample = nn.ConvTranspose2d(nFeat, nChannel, stride=3, kernel_size=17, padding=8)
        #ConvT version
        #self.tail = nn.Sequential(nn.Conv2d(nFeat*5,nFeat*4,kernel_size=1), nn.ConvTranspose2d(nFeat*4, nFeat, kernel_size=3, padding=1, stride=2, output_padding=(1, 1)), nn.Conv2d(nFeat,nChannel,kernel_size=3,padding=1))
        #Pixel Shuffle version
        self.tail = nn.Sequential(nn.Conv2d(nFeat*5,nFeat*4,kernel_size=1), nn.PixelShuffle(2), nn.Conv2d(nFeat,nChannel,kernel_size=3,padding=1))

    def forward(self, x):
       
        x = self.pack(x)
        x = self.entry(x)
        uouts = []
        uouts.append(x)
        x = self.Enhan_unit1(x)
        uouts.append(x)
        x = self.Enhan_unit2(x)
        uouts.append(x)
        x = self.Enhan_unit3(x)
        uouts.append(x)
        x = self.Enhan_unit4(x)
        uouts.append(x)
        x = torch.cat(uouts, dim = 1)

        out = self.tail(x)

        return out  

