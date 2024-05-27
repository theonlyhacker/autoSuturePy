""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.01,inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            #nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1)

    def forward(self, x):
        return F.relu(self.double_conv(x) + self.res(x))
# net = DoubleConv(16,32)
# for i in net.parameters():
#     print(i.nelement())
# total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
# print("Number of parameter: %.6f" % (total))  # 输出

# image = torch.randn((2,1024,160))
# net = nn.Linear(160,256)
# for i in net.parameters():
#     print(i.nelement())
# total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
# print("Number of parameter: %.6f" % (total))  # 输出

# image = torch.randn((2,160,32,32))
# net = nn.Conv2d(160,256,3,1,1)
# for i in net.parameters():
#     print(i.nelement())
# total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
# print("Number of parameter: %.6f" % (total))  # 输出



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2]) 老写法会有警告，下面采用新写法进行实现。

        x1 = F.pad(x1, [torch.div(diffX,2,rounding_mode="floor"), diffX - torch.div(diffX,2,rounding_mode="floor"),
                        torch.div(diffY,2,rounding_mode="floor"), diffY - torch.div(diffY,2,rounding_mode="floor")])


        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)