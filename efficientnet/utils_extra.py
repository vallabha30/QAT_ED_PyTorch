# Author: Zylo117

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):   
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        x = self.quant(x)
        x = self.pool(x)
        x = self.dequant(x)
        return x


class Bn2dWrapper(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-3):  
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.bn = nn.BatchNorm2d(num_features, momentum=momentum, eps=eps) 

    def forward(self, x):
        
        x = self.quant(x)
        x = self.bn(x)
        x = self.dequant(x)
        return x


class UpsampleWrap(nn.Module):
  def __init__(self, scale_factor=2, mode='nearest'):  
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode) 

  def forward(self, x):
      
      x = self.quant(x)
      x = self.up(x)
      x = self.dequant(x)
      return x

class ParameterWrap(nn.Module):
  def __init__(self,abcd=torch.ones(2, dtype=torch.float32) ,requires_grad=True):  
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.param = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=requires_grad)
  def forward(self, x):
      
      x = self.quant(x)
      x = self.param(x)
      x = self.dequant(x)
      return x

class ReluWrap(nn.Module):
  def __init__(self):  
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.relu= nn.ReLU()
  def forward(self, x):
      
      x = self.quant(x)
      x = self.relu(x)
      x = self.dequant(x)
      return x

class ModuleListWrap(nn.Module):
  def __init__(self):  
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.Mlist= nn.ModuleList([])
  def forward(self, x):
      
      x = self.quant(x)
      x = self.Mlist(x)
      x = self.dequant(x)
      return x
