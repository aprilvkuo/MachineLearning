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

class SGD(object):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        print(parameters)
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def zero_grad(self):
        for parameters in self.parameters:
            parameters.weight_grad = 0
            parameters.bias_grad = 0

    def step(self):
        for parameters in self.parameters:
            parameters.v_weight = parameters.v_weight * self.momentum + self.lr * parameters.weight_grad
            parameters.weight -= self.lr * parameters.v_weight
            parameters.v_bias = parameters.v_bias * self.momentum + self.lr * parameters.bias_grad
            parameters.bias -= self.lr * parameters.bgrad


if __name__ == '__main__':
    pass