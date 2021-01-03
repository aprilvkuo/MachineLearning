#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: fc_test.py
@time: 2020/5/5 20:22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    data_train = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root="./data/", transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True, num_workers=2)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True, num_workers=2)

    model = Net(784, 10)
    print(model.parameters())
    # torch.nn.parameter.Paramter
    for item in model.parameters():
        print(type(item))
        print(item.shape)
        # print(item.name)
    exit(1)

    cost = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    n_epochs = 10
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        for data in data_loader_train:
            X_train, y_train = data
            X_train = X_train.view(64, -1)
            # print(X_train.shape)
            # print(y_train.shape)
            # exit(1)
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)
            print("loss is ", loss)
            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()
            # running_loss += loss.data[0]
            # running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0


    #

    output = net(input)

    loss.backward()
    optimizer.step()  # Does the update
    print(net)

