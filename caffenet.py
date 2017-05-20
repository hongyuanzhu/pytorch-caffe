import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from cfg import parse_prototxt

class CaffeConvolution(nn.Module):
    def __init__(self, filters,kernel_size, stride, pad, group):
        super(CaffeConvolution, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.group = group
    def forward(self, x):
        in_filters = x.data.size(1)
        x = nn.Conv2d(in_filters, self.filters, self.kernel_size, self.stride, self.pad,groups=self.group)
        return x
    def __repr__(self):
        return 'Convolution(%d, kernel_size=(%d, %d), stride=(%d, %d), padding=(%d, %d)' % (
                 self.filters, self.kernel_size, self.kernel_size,
                 self.stride, self.stride, self.pad, self.pad)


class CaffeFC(nn.Module):
    def __init__(self, out_channels):
        super(CaffeFC, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        if x.data.dim() == 4:
            x = x.view(nB,-1)
        in_channels = x.data.size(1)
        x = nn.Linear(in_channels, out_channels)
        return x
    def __repr__(self):
        return 'InnerProduct(%d)' % (self.out_channels)

class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info)


    def print_network(self):
        print(self.models)

    def create_network(self, net_info):
        models = nn.Sequential()

        for layer in net_info['layers']:
            name = layer['name']
            name = '%-6s' % (name)
            ltype = layer['type']
            if ltype == 'Data':
                continue
            elif ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                models.add_module(name, CaffeConvolution(filters, kernel_size, stride,pad,group))
            elif ltype == 'ReLU':
                bottom = layer['bottom']
                top = layer['top']
                inplace = (bottom == top)
                models.add_module(name, nn.ReLU(inplace=inplace))
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                models.add_module(name, nn.MaxPool2d(kernel_size, stride))
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                models.add_module(name, CaffeFC(filters))
            elif ltype == 'Softmax':
                models.add_module(name, nn.Softmax())
        return models

if __name__ == '__main__':
    import sys
    protofile = sys.argv[1]
    print(protofile)
    net = CaffeNet(protofile)
    net.print_network()
