'''
通过改变网络参数查看对分类效果的影响

'''
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch import Tensor
from torchvision import datasets, models, transforms
from torch.nn.parameter import Parameter
import numpy as np
#SE
'''
鉴于torch适应于cuda的版本太低导致nn中没有SiLU,出此下策……………………
'''
class SiLU(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1710.05941.pdf]
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

# 注意力机制1
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
# # 注意力机制2
# class sa_layer(nn.Module):
#     """Constructs a Channel Spatial Group module.
#     Args:
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, channel, groups=64):
#         super(sa_layer, self).__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#
#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
#
#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape
#
#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)
#
#         # flatten
#         x = x.reshape(b, -1, h, w)
#
#         return x
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         print('b',b,x.shape)
#         x = x.reshape(b * self.groups, -1, h, w)
#         x_0, x_1 = x.chunk(2, dim=1)
#
#         # channel attention
#         xn = self.avg_pool(x_0)
#         xn = self.cweight * xn + self.cbias
#         xn = x_0 * self.sigmoid(xn)
#
#         # spatial attention
#         xs = self.gn(x_1)
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)
#
#         # concatenate along channel axis
#         out = torch.cat([xn, xs], dim=1)
#         out = out.reshape(b, -1, h, w)
#
#         out = self.channel_shuffle(out, 2)
#         return out
#
# # 注意力机制2 实现2
# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, channel, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)
# 注意力机制3
#SE
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = SiLU()  # alias Swish    nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, x):
        x1 = x[:, 0]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight,padding=1)
        return x1
class filter2D(nn.Module):
    def __init__(self):
        super(filter2D, self).__init__()
        kernel = [[0,-1,0],[-1,4.9,-1],[0,-1,0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight1 = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, x):
        x1 = x[:, 0]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight1,padding=1)
        return x1
#Net Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #定义高斯模块
        self.filter2D = filter2D()
        self.GaussianBlur = GaussianBlur()
        #定义结束
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)#默认 padding=0 stride=1
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(53*53*16, 120)#53x53x16
        self.fc2 = nn.Linear(120, 76)
        self.fc3 = nn.Linear(76, 2)
        #self.SE = SELayer(16)
        self.se = SqueezeExcitation(16,16)
        self.ca = CA_Block(16,53,53)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.filter2D(x)
        x = self.GaussianBlur(x)
        # print(np.array(x.cpu()).shape)

        # print(np.array(x.cpu()).shape)
        # sleep(1000)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 2,stride = 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size= 2,stride = 2)
        x = self.ca(x)
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


