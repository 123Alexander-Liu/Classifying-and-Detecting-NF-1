
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


"""
class  VGGBlock(nn.Module): ## that is a part of model
    def __init__(self,inchannel,middle_channels , outchannel):
        super(VGGBlock,self).__init__()
        ## conv branch
        self.left = nn.Sequential(     ## define a serial of  operation
            nn.Conv2d(inchannel,middle_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels,outchannel,kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel))
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if inchannel != outchannel:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel))
                
    ### get the residual
    def forward(self,x):
        return F.relu(self.left(x) + self.short_cut(x))   """



