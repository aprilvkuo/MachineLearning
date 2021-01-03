#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: optim.py
@time: 2020/5/3 18:16
"""

import numpy as np
from .variable import  Variable


class SGD(object):
    def __init__(self, parameters, lr=0.01, momentum=0.0,
                 weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None

    def zero_grad(self):
        for parameter in self.parameters:
            print(parameter)
            parameter.zero_grad()

    def step(self):
        for parameter in self.parameters:
            d_p = parameter.grad
            if self.weight_decay != 0:
                d_p += self.weight_decay * parameter.data
            if self.momentum != 0:
                if parameter.momentum_buffer is not None:
                    d_p += self.momentum * parameter.momentum_buffer
                parameter.momentum_buffer = d_p
            d_p = -self.lr * d_p
            parameter.add_(d_p)






if __name__ == '__main__':
    pass