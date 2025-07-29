import math

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, h, w, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((h, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, w))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class AP_MP(nn.Module):
    def __init__(self, stride=2):
        super(AP_MP, self).__init__()
        self.sz = stride
        self.gapLayer = nn.AvgPool2d(kernel_size=self.sz, stride=self.sz)
        self.gmpLayer = nn.MaxPool2d(kernel_size=self.sz, stride=self.sz)

    def forward(self, x):
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        byimg = torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        return byimg


class BeFusion(nn.Module):
    def __init__(self, channel, h, w, isFirst=False, isLast=False):
        super(BeFusion, self).__init__()

        self.convTo2 = nn.Conv2d(channel * 2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.h = h
        self.w = w
        self.isFirst=isFirst
        self.coordAttention = CoordAtt(channel, channel, self.h, self.w)
        self.channel = channel

        self.glbamp = AP_MP()
        if isFirst:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(channel * 2 + 1, channel, 1),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
            )
        else:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(channel + 1, channel, 1),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True),
            )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if isFirst:
            self.BConv = nn.Sequential(
                nn.Conv2d(channel, channel, 3, 2, 1, dilation=1, bias=False),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True)
            )
        elif isLast:
            self.BConv = nn.Sequential(
                nn.Conv2d(channel, channel, 3, 1, 1, dilation=1, bias=False),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True)
            )
        else:
            self.BConv = nn.Sequential(
                nn.Conv2d(channel, channel * 2, 3, 2, 1, dilation=1, bias=False),
                nn.BatchNorm2d(channel * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(True)
            )

    def forward(self, r, d, f=None):
        H = torch.cat((r, d), dim=1)
        H_conv = self.sig(self.convTo2(H))
        g = self.global_avg_pool(H_conv)

        ga = g[:, 0:1, :, :]
        gm = g[:, 1:, :, :]

        Ga = r * ga
        Gm = d * gm

        Gm_out = self.coordAttention(Gm)
        res = Gm_out + Ga

        gamp = self.upsample2(self.glbamp(res))
        gamp = gamp / math.sqrt(self.channel)

        if self.isFirst:
            cat = torch.cat((Ga, Gm_out, gamp), dim=1)
        else:
            cat = torch.cat((f, gamp), dim=1)
        cat = self.conv_cat(cat)
        sal = res + cat

        out = self.BConv(sal)

        return sal, out