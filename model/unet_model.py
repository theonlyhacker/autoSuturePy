""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

# from .unet_parts import *
from model.unet_parts import *
from thop import profile

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #n_channels 指的是输入图片的维度   64表示有64个卷积核chaneal
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    net.cuda(0)
    print(net)
    # 利用自带库进行网络参数量计算
    # total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    # print("Number of parameter: %.6f" % (total))  # 输出
    flops_inputs = torch.randn(1, 3, 256, 256).to("cuda:0")
    flops, params = profile(net, inputs=(flops_inputs,))  # 只需要计算一张图的计算量就行
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))