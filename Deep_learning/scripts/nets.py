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

class FC(object):
    def __init__(self, input_size, output_size):
        np.random.seed(7)
        self.__x = None
        self.weight = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)
        self.weight_grad = np.zeros(self.weight.shape)
        self.bias_grad = np.zeros(self.bias.shape)

    def forward(self, x):
        self.__x = x
        out = np.dot(x, self.weight) + self.bias
        return out

    def backward(self, grad):
        delta_x = np.dot(self.weight, grad)
        delta_weight = np.dot(self.__x, grad)
        delta_bias = grad
        self.weight_grad = delta_weight
        self.bias_grad = delta_bias
        return delta_x

    def parameters(self):
        return self


if __name__ == '__main__':
    pass
