#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@contact: aprilvkuo@live.com
@site: 
@software: PyCharm Community Edition
@file: ops.py
@time: 2020/5/5 00:34
"""

from .variable import Variable


def mm(x1: Variable, x2: Variable)->Variable:
    return x1.matmul(x2)

