import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.activate import *


class CBA(nn.Module):
    '''

    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 act_type='SiLU',
                 norm_do=True,
                 act_do=True
                 ):
        super(CBA, self).__init__()
        if bias == 'auto':
            bias = False
        else:
            bias = True
        # # 是否正序/逆序
        # self.order = order

        # 是否执行B+A
        self.norm_do = norm_do
        self.act_do = act_do

        # 创建卷积操作
        self.conv = nn.Conv2d(in_channels, out_channels,
                    kernel_size, stride, padding, dilation, groups, bias)

        # 将卷积层操作参数取出
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # 创建BN层操作
        bn_channels = out_channels
        self.bn = nn.BatchNorm2d(bn_channels)

        # 创建激活函数层
        if act_type == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'SiLU':
            self.act = SiLU()
        elif act_type == 'HS':
            self.act = Hardswish()

    def forward(self, x):

        x = self.conv(x)
        if self.norm_do and self.act_do:
            x = self.bn(x)
            x = self.act(x)
            return x
        elif self.norm_do == False and self.act_do:
            return self.act(x)
        else:
            return x

class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBA(c1, c_, 1, 1)
        self.cv2 = CBA(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))