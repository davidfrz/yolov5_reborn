import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.ConvBlocks import *

class Focus(nn.Module):
    # 将WH的信息编码到 通道 中 WH --> C
    # in_c, out_c, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1):
        super(Focus, self).__init__()
        self.conv = CBA(c1*4, c2, k, s, p, g,norm_do=False,act_do=True)

    def forward(self, x):                              # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]], 1))
# if __name__ == '__main__':
    # net=Focus(3,12)
    # x=torch.randn(1,3,608,608)
    # y=net(x)
    # print(y.shape)
