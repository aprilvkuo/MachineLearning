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

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from nets import FC
from optim import SGD

class MyModel(object):
    def __init__(self):
        self.fc = FC(10, 10)

    def forward(self, x):
        out = self.fc.forward(x)
        return out

    def parameters(self):
        return [self.fc.parameters()]

    def __call__(self, *x):
        return self.forward(*x)


def test():
    model = MyModel()
    optim = SGD(model.parameters())
    for i in range(10):
        optim.zero_grad()
        x = np.random.rand(10, 10)
        y_ = model.forward(x)
        print(y_)


if __name__ == '__main__':
    test()

