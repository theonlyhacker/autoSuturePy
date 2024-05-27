import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from thop import profile

__all__ = ['UNext','UNext_L']

import timm
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import pdb


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv1 = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.dwconv2 = DWConv(hidden_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, D, H, W): # 第四层x.shape =（2，756，160）  (D,H,W)=(6,9,14)
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()  # (2,756,160)->(2,160,756)->(2,160,6,9,14)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant",
                   0)  # 对xn今天上下左右的0填充 变为(2,160,10,13,18)
        xs = torch.chunk(xn, self.shift_size, 1)  # 对(2,160,10,13,18)在维度1分成shift_size=5,分成五块张量。xs有五个张量 每个是(2,32,10,13,18)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # 调整D
        x_cat = torch.cat(x_shift, 1)  # 把从(2,160,6,9,14)拆成的5个(2,32,6,9,14)又在通道维度进行拼叠。
        x_cat = torch.narrow(x_cat, 2, self.pad, D)  # 因为之前有了填充这里有进行缩减回原来的尺寸(2,160,10,13,18)->(2,160,6,9,13)
        x_s = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_s, 4, self.pad, W)
        x_s = x_s.reshape(B, C, D * H * W).contiguous()  # (2,160,756)
        x_shift_r = x_s.transpose(1, 2)  # (2,756,160)
        x = self.fc1(x_shift_r)  # 输入是160 输出是160*mlp_ratio1所以 (2,756,160)->(2,756,160)
        x = self.dwconv1(x, D, H, W) # 深度分离卷积(2,756,160)->(2,756,160)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()  # (2,756,160)->(2,160,756)->(2,160,6,9,14)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # 调整H
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_s = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_s, 4, self.pad, W)
        x_s = x_s.reshape(B, C, D * H * W).contiguous()  # （2，160，6，9，14）->（2，160，756）
        x_shift_c = x_s.transpose(1, 2)  # （2，160，756）->（2，756，160）
        x = self.fc2(x_shift_c)
        x = self.dwconv2(x, D, H, W)  # 深度分离卷积(2,756,160)->(2,756,160)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()  # (2,756,160)->(2,160,756)->(2,160,6,9,14)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # 调整H
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_s = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_s, 4, self.pad, W)
        x_s = x_s.reshape(B, C, D * H * W).contiguous()  # （2，160，6，9，14）->（2，160，756）
        x_shift_c = x_s.transpose(1, 2)  # （2，160，756）->（2，756，160）

        x = self.fc3(x_shift_c)
        x = self.drop(x)
        return x

# class shiftmlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.dim = in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, hidden_features)
#         self.drop = nn.Dropout(drop)
#         self.fc3 = nn.Linear(hidden_features, hidden_features)
#         self.fc4 = nn.Linear(hidden_features, out_features)
#         self.shift_size = shift_size
#         self.pad = shift_size // 2
#         self.norm = nn.LayerNorm(out_features)
#
#         # self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     #     def shift(x, dim):
#     #         x = F.pad(x, "constant", 0)
#     #         x = torch.chunk(x, shift_size, 1)
#     #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#     #         x = torch.cat(x, 1)
#     #         return x[:, :, pad:-pad, pad:-pad]
#
#     def forward(self, x, D, H, W): # 第四层x.shape =（2，756，160）  (D,H,W)=(6,9,14)
#         origin1_x = x
#         B, N, C = x.shape
#         x = self.fc1(x)
#         x = self.act(x)
#
#         xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()  # (2,756,160)->(2,160,756)->(2,160,6,9,14)
#         xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant",
#                    0)  # 对xn今天上下左右的0填充 变为(2,160,10,13,18)
#         xs = torch.chunk(xn, self.shift_size,
#                          1)  # 对(2,160,10,13,18)在维度1分成shift_size=5,分成五块张量。xs有五个张量 每个是(2,32,10,13,18)
#         x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]  # 调整D
#         x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(x_shift, range(-self.pad, self.pad + 1))]  # 调整H
#         x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(x_shift, range(-self.pad, self.pad + 1))]  # 调整W
#
#         x_cat = torch.cat(x_shift, 1)  # 把从(2,160,6,9,14)拆成的5个(2,32,6,9,14)又在通道维度进行拼叠。
#         x_cat = torch.narrow(x_cat, 2, self.pad, D)  # 因为之前有了填充这里有进行缩减回原来的尺寸(2,160,10,13,18)->(2,160,6,9,13)
#         x_s = torch.narrow(x_cat, 3, self.pad, H)
#         x_s = torch.narrow(x_s, 4, self.pad, W)
#         x_s = x_s.reshape(B, C, D * H * W).contiguous()  # (2,160,756)
#         x_shift_r = x_s.transpose(1, 2)  # (2,756,160)
#         x = self.fc2(x_shift_r)  # 输入是160 输出是160*mlp_ratio1所以 (2,756,160)->(2,756,160)
#         x = self.norm(x)
#         x = origin1_x + x
#         origin2_x = x
#
#         x = self.fc3(x)
#         x = self.act(x)
#         x = self.fc4(x)
#         x = self.norm(x)
#         x = origin2_x + x
#
#         x = self.dwconv(x, D, H, W)  # 深度分离卷积(2,756,160)->(2,756,160)
#         x = self.act(x)
#         x = self.norm(x)
#         return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, D ,H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape # (2,756,160)
        x = x.transpose(1, 2).view(B, C, D, H, W) # (2,756,160)->(2,160,756)->(2,160,6,9,13)
        x = self.dwconv(x) # (2,160,6,9,13)->(2,160,6,9,13)
        x = x.flatten(2).transpose(1, 2) # (2,160,6,9,13)->(2,160,756)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size) # shape=(56,56)
        patch_size = to_3tuple(patch_size) # shape=(3,3) 外部传进来的是patch_size=3

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1] # H=18 W=18
        self.num_patches = self.H * self.W # 把一个图片分成了18*18个小补丁。
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[1] // 2))  # kernel_size=3 stride=2 patch_size=1
        self.norm = nn.LayerNorm(embed_dim)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)  # (2,128,11,18,28)->(2,160,6,9,14)       (2,160,6,9,14)->(2,256,3,5,7)
        _, _, D, H, W = x.shape  # (D,H,W)=(6,9,14)                 (D,H,W)=(3,5,7)
        x = x.flatten(2).transpose(1, 2)  # 变为(2,160,756)->(2,756,160)    #(2,256,105)->(2,105,256)
        x = self.norm(x)

        return x, D, H, W  # x图像拉伸后的尺寸 H是图像拉伸前的行 W是图像拉伸前的列 以第四层为例(2,756,160) 32，32


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=1, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv3d(1, 16, 3, stride=1, padding=1)# 输入从3改1 上面num_classes也改为1了
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm3d(16)
        self.ebn2 = nn.BatchNorm3d(32)
        self.ebn3 = nn.BatchNorm3d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv3d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(160)
        self.dbn2 = nn.BatchNorm3d(128)
        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        B = x.shape[0]  # 假设我得输入是(2,1,88,144,224) 这里B=2
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))  # (2,16,44,72,112)
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))  # (2,32,22,36,56)
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))  # (2,128,11,18,28)
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, D, H, W = self.patch_embed3(
            out)  # (2,128,11,18,28)->(2,160,6,9,14)->(2,160,756)->(2,756,160) (D,H,W)=(6,9,14)
        for i, blk in enumerate(self.block1):
            out = blk(out, D, H, W)  # 返回一个调整H,W后的(2,756,160)
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2,
                                                  3).contiguous()  # (2,756,160)->(2,6,9,13,160)->(2,160,6,9,14)
        t4 = out
        ### Bottleneck

        out, D, H, W = self.patch_embed4(out) # (2,160,6,9,14)->(2,256,3,5,7)->(2,256,105)->(2,105,256) (D,H,W)=(3,5,7)
        for i, blk in enumerate(self.block2):
            out = blk(out, D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # (2,256,3,5,7)


        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2),
                                   mode='trilinear'))  # 采用三线性插值上采样 (2,256,3,5,7)->(2,160,6,10,14)
        diffY = torch.tensor([t4.size()[2] - out.size()[2]])
        diffX = torch.tensor([t4.size()[3] - out.size()[3]])
        diffZ = torch.tensor([t4.size()[4] - out.size()[4]])

        out = F.pad(out,
                    [torch.div(diffZ, 2, rounding_mode="floor"), diffZ - torch.div(diffZ, 2, rounding_mode="floor"),
                     torch.div(diffX, 2, rounding_mode="floor"), diffX - torch.div(diffX, 2, rounding_mode="floor"),
                     torch.div(diffY, 2, rounding_mode="floor"), diffY - torch.div(diffY, 2, rounding_mode="floor")])
        out = torch.add(out, t4) # 桥这块用的是直接相加 shape(2,160,6,9,14)
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # shape(2,756,160)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, D, H, W) # (2,756,160)  (D,H,W)=(6,9,14)

        ### Stage 3
        out = self.dnorm3(out)  # (2,756,160)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # (2,756,160)->(2,160,6,9,14)
        out = F.relu(
            F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2),
                          mode='trilinear'))  # 采用三线性插值上采样 (2,160,6,9,14)->(2,128,6,9,14)->(2,128,12,18,28)
        diffY = torch.tensor([t3.size()[2] - out.size()[2]])
        diffX = torch.tensor([t3.size()[3] - out.size()[3]])
        diffZ = torch.tensor([t3.size()[4] - out.size()[4]])

        out = F.pad(out,
                    [torch.div(diffZ, 2, rounding_mode="floor"), diffZ - torch.div(diffZ, 2, rounding_mode="floor"),
                     torch.div(diffX, 2, rounding_mode="floor"), diffX - torch.div(diffX, 2, rounding_mode="floor"),
                     torch.div(diffY, 2, rounding_mode="floor"), diffY - torch.div(diffY, 2, rounding_mode="floor")])

        out = torch.add(out, t3)  # (2,128,11,18,28) (D,H,W)=(11,18,28)
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # (2,128,5544)->(2,5544,128)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, D, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()  # (2,5544,128)->(2,128,11,18,28)
        out = F.relu(
            F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 2), mode='trilinear'))  # (2,32,22,36,56)
        diffY = torch.tensor([t2.size()[2] - out.size()[2]])
        diffX = torch.tensor([t2.size()[3] - out.size()[3]])
        diffZ = torch.tensor([t2.size()[4] - out.size()[4]])
        out = F.pad(out,
                    [torch.div(diffZ, 2, rounding_mode="floor"), diffZ - torch.div(diffZ, 2, rounding_mode="floor"),
                     torch.div(diffX, 2, rounding_mode="floor"), diffX - torch.div(diffX, 2, rounding_mode="floor"),
                     torch.div(diffY, 2, rounding_mode="floor"), diffY - torch.div(diffY, 2, rounding_mode="floor")])

        out = torch.add(out, t2)  # (2,32,22,36,56)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        diffY = torch.tensor([t1.size()[2] - out.size()[2]])
        diffX = torch.tensor([t1.size()[3] - out.size()[3]])
        diffZ = torch.tensor([t1.size()[4] - out.size()[4]])

        out = F.pad(out,
                    [torch.div(diffZ, 2, rounding_mode="floor"), diffZ - torch.div(diffZ, 2, rounding_mode="floor"),
                     torch.div(diffX, 2, rounding_mode="floor"), diffX - torch.div(diffX, 2, rounding_mode="floor"),
                     torch.div(diffY, 2, rounding_mode="floor"), diffY - torch.div(diffY, 2, rounding_mode="floor")])

        out = torch.add(out, t1)  # (2,16,44,72,112)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = self.final(out)
        return self.sig(out)


class UNext_S(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP w less parameters

    def __init__(self, num_classes, input_channels=1, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()
    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
class UNext_L(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=1, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)# 输入从3改1 上面num_classes也改为1了
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(256)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(256)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        print(out.shape)
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        print(out.shape)
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        print(out.shape)

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        #out = self.final(out)
        return self.final(out)



# EOF
if __name__ == '__main__':
    a = torch.randn((2, 1, 88, 144, 224))
    print(a.shape)
    net = UNext(num_classes=2)
    #net = UNext_L(num_classes=2)
    #net = UNext_S(num_classes=2)
    pred = net(a)
    print(pred.shape)
    print(pred)
    total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    print("Number of parameter: %.6f" % (total))  # 输出
    flops, params = profile(net, inputs=(a,))  # 只需要计算一张图的计算量就行
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
