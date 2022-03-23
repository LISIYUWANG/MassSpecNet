'''
1.删减重复层
    第一种删除：
    ca :  max: auc 92.07 acc:84.36
    se :  max: auc 92.50  acc:81.81
          max: auc 92.02  acc: 83.27
    第二种删除：
    ca :



2. 在1基础上层前增加可变卷积

3. 在1基础上在每个module前增加一个可变卷积
    1. 只在真实特征部分加
    2. 只在幻影特征部分加
    3. 两个部分都加

4. 1*3 && 3*1 代替 3*3
    1.改变module **可尝试
    2.改变大块之后的部分

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import ConvOffset2D
from preprocess import GaussianBlur3
from preprocess import DENO
from preprocess import lamplas
from preprocess import GaussianBlur
from preprocess import filter2D
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
#注意力机制2
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
    '''
    foc_type: attention type ：1：ca    2:se
    '''
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,pic_width = 224,if_add_offset = 0,foc_type = 1):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        focus_type = foc_type is not None and foc_type > 0
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
            if focus_type == 2:
                self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
            else:
                self.se = CA_Block(mid_chs,pic_width,pic_width)
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
            # '''
            # 输出注意力之后的图
            # '''
            # print('x ',type(x))
            # print('x ',x.shape)
            # from PIL import Image as img
            # import numpy as np
            # im=x
            # im = np.array(im)
            # print('x ',type(im))
            # print('x ',im.shape)
            #
            # for k in range(120):
            #     print(im[0][k].shape)
            #     ig = im[0][k] * 100
            #     print(ig)
            #     print(ig.shape)
            #     ig = img.fromarray(ig)
            #     ig.show()

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    """"""
    '''
    if_conc: Whether to concat the two networks
    foc_type: Attention type  ca:1  se:2
    del_change_conv: Whether to delete deformable convolution
    '''
    def __init__(self, cfgs, num_classes=2, width=1.0, dropout=0.2,if_conc=True,foc_type = 1,del_change_conv=False):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.if_conc = if_conc
        self.del_change_conv = del_change_conv
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
            for k, exp_size, c, se_ratio, s ,pic_width,if_add_offset in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio,pic_width = pic_width,if_add_offset=if_add_offset,foc_type=foc_type))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.offset22 = ConvOffset2D(input_channel)  # 可变卷积，等会在头试试
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        if self.del_change_conv == False:
            x = self.offset22(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.if_conc == False:
            x = self.classifier(x)
        return x



def ghostnet(if_conc,foc_type,del_change_conv=False):
    """"""
    """
    Constructs a GhostNet model
    if_conc: Whether to concat the two networks
    foc_type: Attention type  ca:1  se:2
    del_change_conv: Whether to delete deformable convolution
    """
    cfgs = [
        # k, t, c, SE, s
        #              ,out_channel,se,stride,pic_width
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
         #[3, 184, 80, 0, 1,0],
         [3, 184, 80, 0, 1,0,0],
         [3, 480, 112, 0, 1,14,0],
         [3, 672, 112, 0, 1,14,0]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2,7,0]],
        [[5, 960, 160, 0, 1,0,0],
         #[5, 960, 160, 0.25, 1,7],
         #[5, 960, 160, 0, 1,0],
         #[5, 960, 160, 0.25, 1,7]
         ]
    ]
    return GhostNet(cfgs=cfgs,if_conc=if_conc,foc_type=foc_type,del_change_conv=del_change_conv)

class NetSimple(nn.Module):
    """"""
    '''
    foc_type: Attention type  ca:1  se:2
    '''
    def __init__(self,foc_type=1):
        super(NetSimple, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(3,6,5)
        # self.conv2 = nn.Conv2d(6, 16, 5)#默认 padding=0 stride=1
        self.conv11 = nn.Conv2d(3, 6, (1,5))
        self.conv12 = nn.Conv2d(6,6,(5,1))
        self.conv21 = nn.Conv2d(6,16,(1,5))
        self.conv22 = nn.Conv2d(16,16,(5,1))
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(53*53*16, 120)#53x53x16
        # self.fc2 = nn.Linear(120, 76)
        # self.fc3 = nn.Linear(76, 2)
        self.focus_type = foc_type
        self.ca = CA_Block(16,53,53)
        self.se = SqueezeExcite(16,16)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv11(x)
        x = F.max_pool2d(F.relu(self.conv12(x)), kernel_size = 2,stride = 2)
        # If the size is a square you can only specify a single number
        x = self.conv21(x)
        x = F.max_pool2d(F.relu(self.conv22(x)),kernel_size= 2,stride = 2)
        if self.focus_type == 1:
            x = self.ca(x)
        elif self.focus_type == 2:
            x = self.se(x)
        x = x.view(-1, self.num_flat_features(x))
        # #print(x.shape)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetConcat(nn.Module):
    """"""
    '''
    foc_type: Attention type  ca:1  se:2
    del_change_conv: Whether to delete deformable convolution
    add_prepro: Whether to add data preprocessing
    '''
    def __init__(self,foc_type=1,del_change_conv = False,add_prepro=True):
        super(NetConcat, self).__init__()
        self.add_prepro = add_prepro
        self.filter2D = filter2D()
        self.GaussianBlur = GaussianBlur()
        self.net1 = ghostnet(True,foc_type=foc_type,del_change_conv=del_change_conv)
        #self.classifier = nn.Linear(46224,2)
        self.fc1 = nn.Linear(46224,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)
        # del self.net1.classifier
        self.net2 = NetSimple(foc_type=foc_type)

    def forward(self, x):
        if self.add_prepro == True:
            x = self.GaussianBlur(x)
            x = self.filter2D(x)
        x1 = self.net1(x)
        #print('x1 ', x1.shape)
        x2 = self.net2(x)
        #print('x2 ',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('x ', x.shape)
        #x = self.classifier(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CEEnetNorui(nn.Module):
    """"""
    '''
    foc_type: Attention type  ca:1  se:2
    del_change_conv: Whether to delete deformable convolution
    add_prepro: Whether to add data preprocessing
    '''
    def __init__(self,foc_type=1,del_change_conv = False,add_prepro=True):
        super(CEEnetNorui, self).__init__()
        self.add_prepro = add_prepro
        self.filter2D = filter2D()
        self.GaussianBlur = GaussianBlur()
        self.net1 = ghostnet(True,foc_type=foc_type,del_change_conv=del_change_conv)
        #self.classifier = nn.Linear(46224,2)
        self.fc1 = nn.Linear(46224,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)
        # del self.net1.classifier
        self.net2 = NetSimple(foc_type=foc_type)

    def forward(self, x):
        # 去除锐化
        # if self.add_prepro == True:
        #     x = self.GaussianBlur(x)
        #     x = self.filter2D(x)
        x1 = self.net1(x)
        #print('x1 ', x1.shape)
        x2 = self.net2(x)
        #print('x2 ',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('x ', x.shape)
        #x = self.classifier(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CEEnetNet1rui(nn.Module):
    """"""
    '''
    foc_type: Attention type  ca:1  se:2
    del_change_conv: Whether to delete deformable convolution
    add_prepro: Whether to add data preprocessing
    '''
    def __init__(self,foc_type=1,del_change_conv = False,add_prepro=True):
        super(CEEnetNet1rui, self).__init__()
        self.add_prepro = add_prepro
        self.filter2D = filter2D()
        self.GaussianBlur = GaussianBlur()
        self.net1 = ghostnet(True,foc_type=foc_type,del_change_conv=del_change_conv)
        #self.classifier = nn.Linear(46224,2)
        self.fc1 = nn.Linear(46224,1600)
        self.fc2 = nn.Linear(1600,760)
        self.fc3 = nn.Linear(760,2)
        # del self.net1.classifier
        self.net2 = NetSimple(foc_type=foc_type)

    def forward(self, x):


        #print('x1 ', x1.shape)
        x2 = self.net2(x)
        # 只对net1锐化
        if self.add_prepro == True:
            x = self.GaussianBlur(x)
            x = self.filter2D(x)
        x1 = self.net1(x)
        #print('x2 ',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('x ', x.shape)
        #x = self.classifier(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CEEnetNet2rui(nn.Module):
    """"""
    '''
    foc_type: Attention type  ca:1  se:2
    del_change_conv: Whether to delete deformable convolution
    add_prepro: Whether to add data preprocessing
    '''

    def __init__(self, foc_type=1, del_change_conv=False, add_prepro=True):
        super(CEEnetNet2rui, self).__init__()
        self.add_prepro = add_prepro
        self.filter2D = filter2D()
        self.GaussianBlur = GaussianBlur()
        self.net1 = ghostnet(True, foc_type=foc_type, del_change_conv=del_change_conv)
        # self.classifier = nn.Linear(46224,2)
        self.fc1 = nn.Linear(46224, 1600)
        self.fc2 = nn.Linear(1600, 760)
        self.fc3 = nn.Linear(760, 2)
        # del self.net1.classifier
        self.net2 = NetSimple(foc_type=foc_type)

    def forward(self, x):
        # print('x1 ', x1.shape)
        x1 = self.net1(x)
        # 只对net2锐化
        if self.add_prepro == True:
            x = self.GaussianBlur(x)
            x = self.filter2D(x)
        x2 = self.net2(x)
        # print('x2 ',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('x ', x.shape)
        # x = self.classifier(x)
        self.fc1before = x
        x = self.fc1(x)
        self.fc2before = x
        x = self.fc2(x)
        self.fc3before = x
        x = self.fc3(x)
        self.fc3after = x
        return x