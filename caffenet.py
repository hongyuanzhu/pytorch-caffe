import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = dict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if block.has_key(key):
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block

    fp = open(protofile, 'r')
    props = dict()
    layers = OrderedDict()
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers[layer['name']] = layer
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = dict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models, self.loss = self.create_network(self.net_info)
        self.modelList = nn.ModuleList()
        for name,model in self.models.items():
            self.modelList.append(model)

    def forward(self, data):
        blobs = OrderedDict()
        blobs['data'] = data
        
        for lname, layer in self.net_info['layers'].items():
            ltype = layer['type']
            if ltype == 'Data' or ltype == 'Accuracy' or ltype == 'SoftmaxWithLoss':
                continue
            tname = layer['top']
            bname = layer['bottom']
            bdata = blobs[bname]
            tdata = self.models[lname](bdata)
            blobs[tname] = tdata
        return blobs.values()[len(blobs)-1]

    def print_network(self):
        print(self.modelList)

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
