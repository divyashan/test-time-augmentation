"""CIFAR10 example for cnn_finetune.
Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb 

from cnn_finetune import make_model
from utils.gpu_utils import restrict_GPU_pytorch
from dataloaders import get_dataloader

parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--dataset', type=str, default='flowers102', metavar='N',
                    help='dataset to finetune to (default: flowers102)')
parser.add_argument('--add', type=bool, default=False, metavar='N',
                    help='include labelled data TTA methods see')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, default='resnet50', metavar='M',
                    help='model name (default: resnet50)')
parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--gpu', type=str, default='0', metavar='D',
                    help='GPU ID')

args = parser.parse_args()
restrict_GPU_pytorch(args.gpu)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def train(model, epoch, optimizer, train_loader, criterion=nn.CrossEntropyLoss()):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    '''Main function to run code in this script'''
    
    
    model_name = args.model_name
    dataset = args.dataset
    if dataset == 'flowers102':
        n_classes = 102
    elif dataset == 'imnet':
        n_classes = 1000

    model = make_model(
        model_name,
        pretrained=True,
        num_classes=n_classes,
        dropout_p=args.dropout_p,
        input_size=(224, 224) if model_name.startswith(('vgg', 'squeezenet')) else None,
    )
    model = model.to(device)
    
    additional_idxs = None
    add = "0"
    batch_size = 128 
    pct = 1.0
    if args.add: 
        additional_idxs = np.load('./' + dataset + '/train_idxs.npy')
        test_idxs = np.load('./' + dataset + '/test_idxs.npy')
        add = "added_data_" + str(pct)
    train_loader = get_dataloader(dataset, 'train', batch_size, True, additional_idxs)

    # If we're training 
    
    test_loader = get_dataloader(dataset, 'val', batch_size, False)
    test(model, test_loader)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)

    # Train
    for epoch in range(1, args.epochs + 1):
        # Decay Learning Rate
        train(model, epoch, optimizer, train_loader)
        test(model, test_loader)
        scheduler.step(epoch)
        if epoch % 5 == 0:
            pdb.set_trace()
            torch.save(model.state_dict(), './saved_models/' + dataset + '/' + model_name + '_' + add + "_" + str(epoch) + '.pth')
if __name__ == '__main__':
    main()
