import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from cfg import parse_prototxt

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x

class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models, self.loss = self.create_network(self.net_info)
        self.models = nn.Sequential(self.models)

    def forward(self, data):
        return self.models(data)

        blobs = OrderedDict()
        blobs['data'] = data
        
        for layer in self.net_info['layers']:
            name = layer['name']
            ltype = layer['type']
            if ltype == 'Data' or ltype == 'Accuracy' or ltype == 'SoftmaxWithLoss':
                continue
            tname = layer['top']
            bname = layer['bottom']
            bottom_data = blobs[bname]
            print(name, bottom_data)
            top_data = self.models[name](bottom_data)
            blobs[tname] = top_data

        return blobs[len(blobs.keys())-1]

    def print_network(self):
        print(self.models)

    def create_network(self, net_info):
        models = OrderedDict()
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()

        blob_channels['data'] = 1
        blob_width['data'] = 28
        blob_height['data'] = 28

        for lname, layer in net_info['layers'].items():
            ltype = layer['type']
            if ltype == 'Data':
                continue
            bname = layer['bottom']
            tname = layer['top']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size, stride,pad,group)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2*pad - kernel_size)/stride + 1
                blob_height[tname] = (blob_height[bname] + 2*pad - kernel_size)/stride + 1
            elif ltype == 'ReLU':
                inplace = (bname == tname)
                models[lname] = nn.ReLU(inplace=inplace)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                models[lname] = nn.MaxPool2d(kernel_size, stride)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]/stride
                blob_height[tname] = blob_height[bname]/stride
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                if blob_width[bname] != 1 or blob_height[bname] != 1:
                    channels = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                    models[lname] = nn.Sequential(FCView(), nn.Linear(channels, filters))
                else:
                    channels = blob_channels[bname]
                    models[lname] = nn.Linear(channels, filters)
                blob_channels[tname] = filters
                blob_width[tname] = 1
                blob_height[tname] = 1
            elif ltype == 'Softmax':
                models[lname] = nn.Softmax()
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
            elif ltype == 'SoftmaxWithLoss':
                loss = nn.CrossEntropyLoss()
        return models, loss

if __name__ == '__main__':
    import sys
    from torch.autograd import Variable
    protofile = sys.argv[1]
    net = CaffeNet(protofile)
    net.print_network()
    for param in net.parameters():
        print(param)
