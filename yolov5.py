import torch
import torch.nn as nn
from nets.Focus import *
from nets.CSPBlocks import *
from nets.ConvBlocks import *
from nets.ResBlocks import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class YOLOv5(nn.Module):
    def __init__(self):
        super(YOLOv5, self).__init__()

        # assert anchors is not None
        cd=2
        wd=3
        self.focus = Focus(3, 64//cd)
        self.conv1 = CBA(64//cd, 128//cd, 3, 2,padding=1)
        self.csp1 = CSP(128//cd, 128//cd, n=3//wd)
        self.conv2 = CBA(128//cd, 256//cd, 3, 2,padding=1)
        self.csp2 = CSP(256//cd, 256//cd, n=9//wd)
        self.conv3 = CBA(256//cd, 512//cd, 3, 2,padding=1)
        self.csp3 = CSP(512//cd, 512//cd, n=9//wd)
        self.conv4 = CBA(512//cd, 1024//cd, 3, 2,padding=1)
        self.spp = SPP(1024//cd, 1024//cd)
        self.csp4 = CSP(1024//cd, 1024//cd, n=3//wd)

        # PANet
        self.conv5 = CBA(1024//cd, 512//cd,1,1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = CSP(1024//cd, 512//cd, n=3//wd)

        self.conv6 = CBA(512//cd, 256//cd,1,1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = CSP(512//cd, 256//cd, n=3//wd)

        self.conv7 = CBA(256//cd, 256//cd, 3, 2,padding=1)
        self.csp7 = CSP(512//cd, 512//cd, n=3//wd)

        self.conv8 = CBA(512//cd, 512//cd, 3, 2,padding=1)
        self.csp8 = CSP(512//cd, 1024//cd, n=3//wd)

        # 改变最后一层通道数 xxx --> 255
        self.ch1=CBA(512,75,norm_do=False,act_do=False)
        self.ch2=CBA(256,75,norm_do=False,act_do=False)
        self.ch3=CBA(128,75,norm_do=False,act_do=False)

    def _build_backbone(self, x):
        x = self.focus(x) # 32*304*304 **********************
        x = self.conv1(x)
        # print(x.shape)
        # print(x.shape)
        x = self.csp1(x)
        # print(x.shape)
        x_p3 = self.conv2(x)  # P3
        # print(x_p3.shape)
        x = self.csp2(x_p3)
        # print(x.shape)
        x_p4 = self.conv3(x)  # P4
        # print(x_p4.shape) #********************  torch.Size([1, 256, 37, 37])
        x = self.csp3(x_p4)
        x_p5 = self.conv4(x)  # P5
        x = self.spp(x_p5)
        x = self.csp4(x)
        return x_p3, x_p4, x_p5, x

    def _build_head(self, p3, p4, p5, feas):
        h_p5 = self.conv5(feas)  # head P5
        x = self.up1(h_p5)
        # print(p4.shape)
        # print(x.shape)
        x_concat = torch.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = torch.cat([x, p3], dim=1)
        x_large = self.csp6(x_concat)

        x = self.conv7(x_large)
        x_concat = torch.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_small = self.csp8(x)

        # 将通道数全部改为255
        x_small = self.ch1(x_small)
        x_medium = self.ch2(x_medium)
        x_large = self.ch3(x_large)

        return x_small, x_medium, x_large

    def forward(self, x):
        p3, p4, p5, feas = self._build_backbone(x)
        xs, xm, xl = self._build_head(p3, p4, p5, feas)
        return xs, xm, xl

if __name__ == "__main__":
    a = torch.randn([1, 3, 608, 608]).to(device)  # 原始 512

    anchors = [[4, 6,  5, 12,  8, 8],
               [13, 12,  8, 20,  13, 31],
               [32, 20,  18, 42,  28, 59]]
    model = YOLOv5().to(device)

    # print(model.parameters())
    o = model(a)
    for a in o:
        print(a.shape)
    print('this is the output of yolov5 to be sent to Detect layer')


