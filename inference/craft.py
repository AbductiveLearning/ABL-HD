# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet50 import init_weights, unetp_resnet50


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=True, amp=True):
        super(CRAFT, self).__init__()

        self.amp = amp

        """ Base network """
        self.basenet = unetp_resnet50(pretrained)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        if self.amp:
            with torch.cuda.amp.autocast():
                feature = self.basenet(x)
                y = self.conv_cls(feature)

                return y.permute(0, 2, 3, 1), feature
        else:
            feature = self.basenet(x)
            y = self.conv_cls(feature)

            return y.permute(0, 2, 3, 1), feature
