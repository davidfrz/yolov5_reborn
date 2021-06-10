import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.ConvBlocks import *


class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,expansion=0.5):
        super(ResBlock,self).__init__()

        c_mid=int(out_c * expansion)
        self.conv1 = CBA(in_c,c_mid,1,1)
        self.conv2 = CBA(c_mid,out_c,3,1,padding=1)

        # self.add = in_c == out_c

    def forward(self,x):
        # return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))
        return x + self.conv2(self.conv1(x))


if __name__ == '__main__':
    net=ResBlock(16,16)
    x=torch.randn(1,16,608,608)
    y=net(x)
    print(y.shape)
