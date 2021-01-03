#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: nets.py
@time: 2020/5/3 14:20
"""

import numpy as np
from .variable import Variable
from . import ops


class Linear(object):
    def __init__(self, input_size, output_size):
        self.__x = None
        self.weight = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)
        self.weight = Variable(self.weight)
        self.bias = Variable(self.bias)

    def __call__(self, x: Variable)->Variable:
        self.__x = x
        # print(x, self.weight)
        out = ops.mm(x, self.weight) + self.bias
        return out

    # def __call__(self, *args, **kwargs):
    #     return self.forward(args)

    def backward(self, grad: np.array)->np.array:
        delta_x = np.dot(self.weight.data, grad)
        delta_weight = np.dot(self.__x.data, grad)
        delta_bias = grad
        self.weight.grad = delta_weight
        self.bias.grad = delta_bias
        return delta_x

    def parameters(self):
        return [self.weight, self.bias]


# if __name__ == '__main__':
#     pass
