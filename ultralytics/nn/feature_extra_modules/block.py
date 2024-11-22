#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union
from einops import rearrange
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .attention import *

__all__ = ['C2f_EMBC', 'CSPStage']


'Mobile Inverted Bottleneck Convolution (MBConv)模块'  'SENet' 'MobileNetV1、2、3、4'
class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1) -> None:
        # inc 输入通道数 | ouc 输出通道数 | e 扩展因子
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)    # 组卷积
        self.effective_se = EffectiveSEModule(midc)     # Squeeze-and-Excitation (SE) 模块
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)          # 2D dropout层 ：用于在卷积层后应用 dropout 防止过拟合
        self.add = shortcut and inc == ouc

    def forward(self, x):
        return x + self.dropout(
            self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))) if self.add else self.dropout(
            self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))


class C2f_EMBC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut) for _ in range(n))


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


'Cross Stage Partial (CSP)' 'CSPNet'
class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, block_fn='BasicBlock_3x3_Reverse', ch_hidden_ratio=1.0, act='silu', spp=False):
        # block_fn 使用的块类型 | 默认激活函数类型silu | 空间金字塔池化SPP
        super(CSPStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()    # 初始化一个顺序容器nn.Sequential 用于存储多个卷积块

        next_ch_in = ch_mid     # 初始化下一个卷积块的输入通道数
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(str(i), BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid     # 更新下一个卷积块的输入通道数
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
