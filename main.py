from __future__ import print_function
import argparse
import os
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from caffenet import CaffeNet, parse_prototxt

#import dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('mode', help='train/time/test')
parser.add_argument('--gpu', help='gpu ids e.g "0,1,2,3"')
parser.add_argument('--solver', help='the solver prototxt')
parser.add_argument('--model', help='the network definition prototxt')
parser.add_argument('--snapshot', help='the snapshot solver state to resume training')
parser.add_argument('--weights', help='the pretrained weight')

args = parser.parse_args()
if args.mode == 'time':

    protofile  = args.model
    net_info   = parse_prototxt(protofile)
    model      = CaffeNet(protofile)
    batch_size = 64

    model.print_network()
    model.eval()

    niters = 50
    total = 0
    for i in range(niters):
        x = torch.rand(batch_size, 1, 28, 28)
        x = Variable(x)
        t0 = time.time()
        output = model(x)
        t1 = time.time()
        print('iteration %d: %fs' %(i, t1-t0))
        total = total + (t1-t0)
    print('total time %fs, %fs per batch, %fs per sample' % (total, total/niters, total/(niters*batch_size)))
    
elif args.mode == 'train':
    solver        = parse_prototxt(args.solver)
    protofile     = solver['net']
    net_info      = parse_prototxt(protofile)
    
    batch_size    = 64
    num_workers   = 1
    learning_rate = 0.01
    momentum      = 0.9
    weight_decay  = 1e-4
    log_interval  = 20
    start_epoch   = 0
    end_epoch     = 100
    
    torch.manual_seed(int(time.time()))
    if args.gpu:
        torch.cuda.manual_seed(int(time.time()))
    
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if args.gpu else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    model = CaffeNet(protofile)
    model.print_network()
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = torch.nn.DataParallel(model).cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    if args.weights:
        state = torch.load(args.weights)
        start_epoch = state['epoch']+1
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded state %s' % (args.weights))
    
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = model.loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
    
        savename = '%06d.pth.tar' % (epoch)
        print('save state %s' % (savename))
        state = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(state, savename)
    
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += model.loss(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
    
        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    
    if args.weights:
        test(start_epoch)
    
    for epoch in range(start_epoch, end_epoch):
        train(epoch)
        test(epoch)
