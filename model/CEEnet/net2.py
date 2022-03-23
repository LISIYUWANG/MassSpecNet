
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import ConvOffset2D
from preprocess import GaussianBlur
from preprocess import filter2D
from preprocess import lamplas
import numpy as np
import cv2
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

# 注意力机制
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, if_add_offset = 0,relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        self.if_add_offset = if_add_offset
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        if self.if_add_offset > 0:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(

                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

        else:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, ca_ratio=0.,pic_width = 224,if_add_offset = 0):
        super(GhostBottleneck, self).__init__()
        has_ca = ca_ratio is not None and ca_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,if_add_offset= if_add_offset)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_ca:
            self.ca = CA_Block(mid_chs,pic_width,pic_width)
        else:
            self.ca = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,if_add_offset= if_add_offset)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.ca is not None:
            #print('x ',x.shape)
            x = self.ca(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x
class GhostBottleneck2(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,pic_width = 224,if_add_offset = 0):
        super(GhostBottleneck2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,if_add_offset= if_add_offset)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,if_add_offset= if_add_offset)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            #print('x ',x.shape)
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=2, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, ca_ratio, s ,pic_width,if_add_offset in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    ca_ratio=ca_ratio,pic_width = pic_width,if_add_offset=if_add_offset))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.offset22 = ConvOffset2D(input_channel)
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        #---------可变形卷积----------------------
        x = self.offset22(x)
        x = self.conv_head(x)
        #-----------------------------------
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
class GhostNet2(nn.Module):
    def __init__(self, cfgs, num_classes=2, width=1.0, dropout=0.2):
        super(GhostNet2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, ca_ratio, s ,pic_width,if_add_offset in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    ca_ratio=ca_ratio,pic_width = pic_width,if_add_offset=if_add_offset))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        #self.offset22 = ConvOffset2D(input_channel)
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
class GhostNet3(nn.Module):
    def __init__(self, cfgs, num_classes=2, width=1.0, dropout=0.2):
        super(GhostNet3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck2
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s ,pic_width,if_add_offset in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio,pic_width = pic_width,if_add_offset=if_add_offset))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.offset22 = ConvOffset2D(input_channel)
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        #---------可变形卷积----------------------
        x = self.offset22(x)
        x = self.conv_head(x)
        #-----------------------------------
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
class GhostNet4(nn.Module):
    def __init__(self, cfgs, num_classes=2, width=1.0, dropout=0.2):
        super(GhostNet4, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, ca_ratio, s ,pic_width,if_add_offset in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    ca_ratio=ca_ratio,pic_width = pic_width,if_add_offset=if_add_offset))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.offset22 = ConvOffset2D(input_channel)
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        #---------可变形卷积----------------------
        x = self.offset22(x)
        x = self.conv_head(x)
        #-----------------------------------
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
# 复杂网络调用
def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1,0,0]],
        # stage2
        [[3, 48, 24, 0, 2,0,0]],
        [[3, 72, 24, 0, 1,0,0]],
        # stage3
        [[5, 72, 40, 0, 2,28,0]],
        [[5, 120, 40, 0.25, 1,28,0]],
        # stage4
        [[3, 240, 80, 0, 2,0,0]],
        [[3, 200, 80, 0, 1,0,0],
         [3, 184, 80, 0, 1,0,0],
         [3, 480, 112, 0, 1,14,0],
         [3, 672, 112, 0, 1,14,0]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2,7,0]],
        [[5, 960, 160, 0, 1,0,0]]
    ]
    return GhostNet(cfgs, **kwargs)
def ghostnet2(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1,0,0]],
        # stage2
        [[3, 48, 24, 0, 2,0,0]],
        [[3, 72, 24, 0, 1,0,0]],
        # stage3
        [[5, 72, 40, 0, 2,28,0]],
        [[5, 120, 40, 0.25, 1,28,0]],
        # stage4
        [[3, 240, 80, 0, 2,0,0]],
        [[3, 200, 80, 0, 1,0,0],
         [3, 184, 80, 0, 1,0,0],
         [3, 480, 112, 0, 1,14,0],
         [3, 672, 112, 0, 1,14,0]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2,7,0]],
        [[5, 960, 160, 0, 1,0,0]]
    ]
    return GhostNet2(cfgs, **kwargs)
def ghostnet3(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1,0,0]],
        # stage2
        [[3, 48, 24, 0, 2,0,0]],
        [[3, 72, 24, 0, 1,0,0]],
        # stage3
        [[5, 72, 40, 0, 2,28,0]],
        [[5, 120, 40, 0.25, 1,28,0]],
        # stage4
        [[3, 240, 80, 0, 2,0,0]],
        [[3, 200, 80, 0, 1,0,0],
         [3, 184, 80, 0, 1,0,0],
         [3, 480, 112, 0, 1,14,0],
         [3, 672, 112, 0, 1,14,0]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2,7,0]],
        [[5, 960, 160, 0, 1,0,0]]
    ]
    return GhostNet3(cfgs, **kwargs)
def ghostnet4(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1,0,0]],
        # stage2
        [[3, 48, 24, 0, 2,0,0]],
        [[3, 72, 24, 0, 1,0,0]],
        # stage3
        [[5, 72, 40, 0, 2,28,0]],
        [[5, 120, 40, 0.25, 1,28,0]],
        # stage4
        [[3, 240, 80, 0, 2,0,0]],
        [[3, 200, 80, 0, 1,0,0],
         [3, 184, 80, 0, 1,0,0],
         [3, 480, 112, 0, 1,14,0],
         [3, 672, 112, 0, 1,14,0]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2,7,0]],
        [[5, 960, 160, 0, 1,0,0]]
    ]
    return GhostNet4(cfgs, **kwargs)
# 简单网络1
class NetSimple(nn.Module):
    def __init__(self):
        super(NetSimple, self).__init__()
        self.GaussianBlur = GaussianBlur()
        self.filter2D = filter2D()

        self.conv11 = nn.Conv2d(3, 6, (1,5))
        self.conv12 = nn.Conv2d(6,6,(5,1))
        self.conv21 = nn.Conv2d(6,16,(1,5))
        self.conv22 = nn.Conv2d(16,16,(5,1))
        self.ca = CA_Block(16,53,53)
    def forward(self, x):
        x = self.GaussianBlur(x)
        x = self.filter2D(x)
        x = self.conv11(x)
        x = F.max_pool2d(F.relu(self.conv12(x)), kernel_size = 2,stride = 2)
        x = self.conv21(x)
        x = F.max_pool2d(F.relu(self.conv22(x)),kernel_size= 2,stride = 2)
        x = self.ca(x)
        x = x.view(-1, self.num_flat_features(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# 简单网络2
class NetSimple2(nn.Module):
    def __init__(self):
        super(NetSimple2, self).__init__()
        self.GaussianBlur = GaussianBlur()
        self.filter2D = filter2D()

        self.conv11 = nn.Conv2d(3, 6, (1,5))
        self.conv12 = nn.Conv2d(6,6,(5,1))
        self.conv21 = nn.Conv2d(6,16,(1,3))
        self.conv22 = nn.Conv2d(16,16,(3,1))
        self.ca = CA_Block(16,54,54)
    def forward(self, x):
        x = self.GaussianBlur(x)
        x = self.filter2D(x)
        x = self.conv11(x)
        x = F.max_pool2d(F.relu(self.conv12(x)), kernel_size = 2,stride = 2)
        x = self.conv21(x)
        x = F.max_pool2d(F.relu(self.conv22(x)),kernel_size= 2,stride = 2)

        x = self.ca(x)
        x = x.view(-1, self.num_flat_features(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class NetSimple3(nn.Module):
    def __init__(self):
        super(NetSimple3, self).__init__()
        self.GaussianBlur = GaussianBlur()
        self.filter2D = filter2D()

        self.conv11 = nn.Conv2d(3, 6, (1,5))
        self.conv12 = nn.Conv2d(6,6,(5,1))
        self.conv21 = nn.Conv2d(6,16,(1,3))
        self.conv22 = nn.Conv2d(16,16,(3,1))
        self.se = SqueezeExcite(16,0.25)
    def forward(self, x):
        x = self.GaussianBlur(x)
        x = self.filter2D(x)
        x = self.conv11(x)
        x = F.max_pool2d(F.relu(self.conv12(x)), kernel_size = 2,stride = 2)
        x = self.conv21(x)
        x = F.max_pool2d(F.relu(self.conv22(x)),kernel_size= 2,stride = 2)

        x = self.se(x)
        x = x.view(-1, self.num_flat_features(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class NetSimple4(nn.Module):
    def __init__(self):
        super(NetSimple4, self).__init__()
        # self.filter2D = filter2D()
        # self.GaussianBlur = GaussianBlur()
        self.conv11 = nn.Conv2d(3, 6, (1,5))
        self.conv12 = nn.Conv2d(6,6,(5,1))
        self.conv21 = nn.Conv2d(6,16,(1,3))
        self.conv22 = nn.Conv2d(16,16,(3,1))
        self.ca = CA_Block(16,54,54)
    def forward(self, x):

        x = self.conv11(x)
        x = F.max_pool2d(F.relu(self.conv12(x)), kernel_size = 2,stride = 2)
        x = self.conv21(x)
        x = F.max_pool2d(F.relu(self.conv22(x)),kernel_size= 2,stride = 2)

        x = self.ca(x)
        x = x.view(-1, self.num_flat_features(x))
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#--------以下5个需要跑-------------------------------------

# 网络合并  最终模型
class NetConcat(nn.Module):
    def __init__(self):
        super(NetConcat, self).__init__()
        self.net1 = ghostnet()
        self.net2 = NetSimple2()
        self.fc1 = nn.Linear(47936,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)

    def forward(self, x):

        x1 = self.net1(x)
        x2 = self.net2(x)

        x = torch.cat([x1, x2], dim=1)
        #print('concat,',x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
# 不加可变形卷积
class NetConcat2(nn.Module):
    def __init__(self):
        super(NetConcat2, self).__init__()
        self.net1 = ghostnet2()
        self.net2 = NetSimple2()
        self.fc1 = nn.Linear(47936,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)

    def forward(self, x):

        x1 = self.net1(x)
        x2 = self.net2(x)

        x = torch.cat([x1, x2], dim=1)
        #print('concat,',x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
#注意力为SE
class NetConcat3(nn.Module):
    def __init__(self):
        super(NetConcat3, self).__init__()
        self.net1 = ghostnet3()
        self.net2 = NetSimple3()
        self.fc1 = nn.Linear(47936,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)

    def forward(self, x):

        x1 = self.net1(x)
        x2 = self.net2(x)

        x = torch.cat([x1, x2], dim=1)
        #print('concat,',x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
#不加预处理
class NetConcat4(nn.Module):
    def __init__(self):
        super(NetConcat4, self).__init__()
        self.net1 = ghostnet()
        self.net2 = NetSimple4()
        self.fc1 = nn.Linear(47936,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)

    def forward(self, x):

        x1 = self.net1(x)
        x2 = self.net2(x)

        x = torch.cat([x1, x2], dim=1)
        #print('concat,',x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
#单边复杂网络
#模型直接调用 model=ghostnet4()

if __name__ == '__main__':


    print(NetConcat())
    print('-----------------------------------------------------------------------------------------')
    print(NetConcat2())
    print('-----------------------------------------------------------------------------------------')
    print(NetConcat3())
    print('-----------------------------------------------------------------------------------------')
    print(NetConcat4())
    print('-----------------------------------------------------------------------------------------')
    print(ghostnet4())