import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.ResBlocks import *


class CSP(nn.Module):
    def __init__(self, in_c, out_c, n=1, expansion=0.5):
        super(CSP, self).__init__()
        mid_c = int(out_c * expansion)

        self.conv1 = CBA(in_c, mid_c, 1, 1)
        self.conv2 = nn.Conv2d(in_c, mid_c, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(mid_c, mid_c, 1, 1, bias=False)

        self.conv4 = CBA(2 * mid_c, out_c, 1, 1)
        self.bn = nn.BatchNorm2d(2 * mid_c)
        self.act = SiLU()
        self.R = nn.Sequential(
            *[ResBlock(mid_c, mid_c, expansion=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.conv3(self.R(self.conv1(x)))
        y2 = self.conv2(x)

        return self.conv4(self.act(self.bn(torch.cat([y1, y2], dim=1))))


class CSPX(nn.Module):
    def __init__(self, in_c, out_c, n=1, expansion=0.5):
        super(CSPX, self).__init__()
        mid_c = int(out_c * expansion)

        self.conv1 = CBA(in_c, mid_c, 1, 1)
        self.conv2 = CBA(in_c, mid_c, 1, 1, bias=False, norm_do=False, act_do=False)
        self.conv3 = CBA(mid_c, mid_c, 1, 1, bias=False, norm_do=False, act_do=False)

        self.conv4 = CBA(2 * mid_c, out_c, 1, 1)
        self.bn = nn.BatchNorm2d(2 * mid_c)
        self.act = SiLU()
        self.R = nn.Sequential(
            *[CBA(mid_c, mid_c, 1, 1) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.conv3(self.R(self.con1(x)))
        y2 = self.conv2(x)

        return self.conv4(self.act(self.bn(torch.cat([y1, y2], dim=1))))

if __name__ == '__main__':
    net=CSP(3,16)
    x=torch.randn(1,3,608,608)
    y=net(x)
    print(y.shape)