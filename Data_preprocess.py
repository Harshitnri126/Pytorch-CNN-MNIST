#!/usr/bin/env python
# coding: utf-8
import pickle
import torch

class preprocess():
    def __init__(self,data):
        self.data=data
    
    def train_data(self):
        train=self.data[0]
        x_train=torch.tensor(train[0])
        y_train=torch.tensor(train[1])
        return x_train.view(-1,1,28,28),y_train
    
    def test_data(self):
        test=self.data[1]
        x_test=torch.tensor(test[0])
        y_test=torch.tensor(test[1])
        return x_test.view(-1,1,28,28),y_test      






