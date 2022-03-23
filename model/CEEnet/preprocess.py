import torch.nn as nn
import torch
import torch.nn.functional as F

class filter2D(nn.Module):
    def __init__(self):
        super(filter2D, self).__init__()
        kernel = [[0,-1,0],[-1,4.9,-1],[0,-1,0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight,padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight,padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight,padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
# 高斯降噪
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
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight,padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight,padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight,padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

# 高斯降噪
class GaussianBlur2(nn.Module):
    def __init__(self):
        super(GaussianBlur2, self).__init__()
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
          [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
          [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
          [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
          [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

# 高斯降噪
class GaussianBlur3(nn.Module):
    def __init__(self):
        super(GaussianBlur3, self).__init__()
        kernel =[
            [1/16,2/16,1/16],
            [2/16,4/16,2/16],
            [1/16,2/16,1/16]
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
# 拉普拉斯算子锐化
class lamplas(nn.Module):
    def __init__(self):
        super(lamplas, self).__init__()
        kernel = [[0,-1,0],[-1,5,-1],[0,-1,0]]
        '''
        kernel2 = [
                    [0,0,-1,0,0],
                    [0,2,2,2,0],
                    [-1,2,4.9,2,-1],
                    [0,2,2,2,0],
                    [0,0,-1,0,0]]
        
        '''
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class lamplas2(nn.Module):
    def __init__(self):
        super(lamplas2, self).__init__()
        kernel = [ [0,0,-1,0,0],
                    [0,2,2,2,0],
                    [-1,2,4.9,2,-1],
                    [0,2,2,2,0],
                    [0,0,-1,0,0]]
        '''
        kernel2 = [
                    [-1,-1,-1,-1,-1],
                    [-1,2,2,2,-1],
                    [-1,2,8,2,-1],
                    [-1,2,2,2,-1],
                    [-1,-1,-1,-1,-1]]

        '''
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
class DENO(nn.Module):
    def __init__(self):
        super(DENO, self).__init__()
        kernel =  [[-1,0,1],
                 [-2,0,2],
                [-1,0,1]]
        '''
        [[-1,-2,-1],
         [0,0,0],
         [1,2,1]]
         
         [[-1,0,1],
         [-2,0,2],
         [-1,0,1]]
         
         [[1,0,-1],
         [2,0,-2],
         [1,0,-1]]
         
         [[1,0,-1],
         [1,0,-1],
         [1,0,-1]]

        '''
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
        return x

