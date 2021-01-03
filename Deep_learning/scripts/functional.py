#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: functional.py
@time: 2020/5/6 00:57
"""

import numpy as np


def softmax(input, dim=0):
    e_x = np.exp(input)
    e_x_sum = np.sum(e_x, axis=dim, keepdims=True)
    return e_x / e_x_sum


def nll_loss(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    class_num = input.shape[1]
    one_hot = np.eye(class_num)[target.reshape(-1)].T
    return -np.sum(input * one_hot) / input.shape[0]


def cross_entropy(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    return nll_loss(np.log(softmax(input, 1)), target, weight, size_average, ignore_index, reduce)



