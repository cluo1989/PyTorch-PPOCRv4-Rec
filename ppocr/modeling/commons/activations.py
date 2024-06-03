'''
Author: Cristiano-3 chunanluo@126.com
Date: 2024-05-23 10:00:27
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2024-05-23 15:21:43
FilePath: /PyTorch-PPOCRv4-Rec/ppocr/modeling/commons/activations.py
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        #return F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.gelu(x)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        self.act_type = act_type.lower()

        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'hardsigmoid':
            self.act = HardSigmoid(inplace=inplace)
        elif act_type == 'hardswish':
            self.act = HardSwish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)

