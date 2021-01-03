#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: loss.py
@time: 2020/5/5 21:34
"""

import numpy as np
import torch
import functional as F
from variable import Variable
# cost = torch.nn.CrossEntropyLoss()
# loss = cost(outputs, y_train)


class CrossEntropyLoss(object):
    def __init__(self):
        pass

    def __call__(self, input, target):
        return F.cross_entropy(input.data, target.data)


if __name__ == '__main__':
    np.random.seed(7)
    y = np.array([0, 1, 2])
    x = np.random.rand(3, 3)
    x_1 = F.softmax(x, 1)
    x_1 = np.log(x_1)
    # print(x_1)
    print(F.nll_loss(x_1, y))

    loss = torch.nn.NLLLoss()
    x_v = torch.autograd.Variable(torch.FloatTensor(x))
    x_1 = torch.nn.functional.softmax(x_v, 1)
    x_1 = torch.log(x_1)
    # print(x_1)
    print(loss(x_1, torch.autograd.Variable(torch.LongTensor(y))))

    loss = torch.nn.CrossEntropyLoss()
    print(loss(x_v, torch.autograd.Variable(torch.LongTensor(y))))

    loss = CrossEntropyLoss()
    print(loss(Variable(x), Variable(y)))

