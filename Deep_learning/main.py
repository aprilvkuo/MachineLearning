#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: main.py
@time: 2020/5/3 20:37
"""

# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import numpy as np
import sys
import os
from scripts.nets import Linear
from scripts.optim import SGD
from scripts.variable import Module, Variable
from scripts.loss import CrossEntropyLoss

np.set_printoptions(threshold=10)


class MyModel(Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc = Linear(input_size, output_size)
        self.init_paraments()

    def init_paraments(self):
        self._parameters.extend(self.fc.parameters())

    def forward(self, x):
        # out = self.fc.forward(x)
        out = self.fc(x)
        return out

    def parameters(self):
        return self._parameters

    def __call__(self, *x):
        return self.forward(*x)


def test():
    model = MyModel(10, 10)
    optim = SGD(model.parameters())
    loss = CrossEntropyLoss()
    for i in range(10):

        x = Variable(np.random.rand(100, 10))
        y = Variable(np.random.randint(0, 10, 100))
        y_pred = model.forward(x)
        cost = loss(y_pred, y)
        optim.zero_grad()

        print(y)


if __name__ == '__main__':
    test()


