#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: lr.py
@time: 2020/5/3 10:14
"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from liner_regression import LinerRegression


class LR(LinerRegression):
    def __init__(self, feature_size)->None:
        np.random.seed(7)
        self._w = np.random.rand(feature_size)
        self._b = np.random.rand(feature_size)

    @staticmethod
    def __sigmod_fuc(x):
        """
        sigmod 函数， 单调递增
        :param x: 
        :return: 
        """
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x: np.array)->np.array:
        """
        
        :param x: 
        :return: y为1的概率
        """
        y = self._w * x + self._b
        out = LR.__sigmod_fuc(y)
        return out


def test_lr():
    feature_size = 10
    x = np.random.randint(5, size=10)
    model = LR(feature_size)
    print(model.forward(x))


if __name__ == '__main__':
    test_lr()