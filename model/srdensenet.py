import torch
import torch.nn as nn

def make_model(args, parent=False):
    return SRDenseNet()

class DenseBlock(nn.Module):
    
    def __init__(self, num_layers, in_channels, out_channels, kernel_size):
        super(DenseBlock, self).__init__()
        modules = [[nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2), nn.ReLU()]]
        modules.extend([[nn.Conv2d(out_channels*i, out_channels, kernel_size, padding=(kernel_size-1)//2), nn.ReLU()] for i in range(1, num_layers)])
        
        self.modules1 = nn.ModuleList()
        for module in modules:
            self.modules1.append(nn.Sequential(*module))
    
    def forward(self, x):
        l_list = []    
        for layer in self.modules1:
            x = layer(x)
            l_list.append(x)
            x = torch.cat(l_list, dim=1)
        return torch.cat(l_list, dim=1)





class SRDenseNet(nn.Module):
    def __init__(self):
        super(SRDenseNet, self).__init__()
        modules_head = [nn.Conv2d(1,256,kernel_size=4,stride=2, padding=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                        nn.PReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, padding=1),
                        nn.ReLU()
                    ]

        modules_body = [DenseBlock(8, 128, 16, 3) for i in range(8)]

        self.bottleneck = nn.Sequential(nn.Conv2d(128*9, 256, 3, padding=1),nn.ReLU())

        modules_tails = [nn.PixelShuffle(2),
                         nn.Conv2d(64, 3, kernel_size=3, padding = 1)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tails)

    def forward(self, x):
        l_list = []
        x = self.head(x)
        l_list.append(x)
        for layer in self.body:
            x = layer(x)
            l_list.append(x)
      
        x = torch.cat(l_list, dim=1)
        x = self.bottleneck(x)
        x = self.tail(x)
        
        return x

        
