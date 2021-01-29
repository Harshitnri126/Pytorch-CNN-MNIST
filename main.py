#!/usr/bin/env python
# coding: utf-8

import pickle
import torch
from Data_preprocess import preprocess
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



def train(args,model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args,model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser=argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batchsize', '-tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', '-sm', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args=parser.parse_args()
    print("=====================================")
    print("# train model with CPU")
    print('# minibatch-size: {0}'.format(args.batchsize))
    print('# epochs: {0}'.format(args.epochs))
    print('# learning rate: {0}'.format(args.lr))
    print("=====================================")
    
    device=torch.device("cpu")
    torch.manual_seed(args.seed)

    with open("mnist.pkl", "rb") as input_file:
        data = pickle.load(input_file,encoding="latin-1")

    x_train,y_train=preprocess(data).train_data()
    x_test,y_test=preprocess(data).test_data()
    train_ds=TensorDataset(x_train,y_train)
    train_dl=DataLoader(train_ds,batch_size=64,shuffle=True)
    test_ds=TensorDataset(x_test,y_test)
    test_dl=DataLoader(test_ds,batch_size=64,shuffle=True)
    
    model=Net()
    model=model.to(device)
    args.log_interval=10
    
    optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)
    
    for epoch in range(1,args.epochs+1):
        train(args,model,device,train_dl,optimizer,epoch)
        test(args,model,device,test_dl)
        

if __name__=='__main__':
    main()
