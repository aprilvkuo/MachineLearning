#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: variable.py
@time: 2020/5/4 23:37
"""

import numpy as np
import torch

# a = torch.nn.Parameter(torch.Tensor([10]))
# torch.FloatTensor


class Variable(object):
    def __init__(self, data=None):
        self.data = data
        # if len(args) > 0:
        #     self.data = np.random.rand(*args)
        self.grad = None
        self.momentum_buffer = None
        self._dep_v = []

    def zero_grad(self):
        self.grad = None
        self.momentum_buffer = None

    def matmul(self, other):
        res = np.dot(self.data, other.data)
        res_var = Variable(res)
        res_var._dep_v.append(self)

        return res_var

    def __add__(self, other):
        res = self.data + other.data
        res_var = Variable(res)
        return res_var

    def add_(self, other: np.array):
        self.data += other
        return

    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def self(self):
        return self

class Module(object):
    def __init__(self):
        self._parameters = []


if __name__ == '__main__':
    pass

