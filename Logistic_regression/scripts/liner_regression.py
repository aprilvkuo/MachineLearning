#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: liner_regression.py
@time: 2020/5/3 10:39
"""

import numpy as np


class LinerRegression(object):
    def __init__(self, feature_size)->None:
        np.random.seed(7)
        self._w = np.random.rand(feature_size)
        self._b = np.random.rand(feature_size)

    def forward(self, x: np.array)->np.array:
        """
        sigmod 函数
        逻辑回归问题是一个01二分的回归问题
        sigmod 函数值为预测值为0的概率。
        :param x: 
        :return: 
        """
        out = np.exp(self._w * x + self._b)
        out = 1.0 / (1.0 + out)
        return out


def test():
    feature_size = 10
    x = np.random.randint(100, size=10)
    model = LinerRegression(feature_size)
    print(model.forward(x))


if __name__ == '__main__':
    test()
