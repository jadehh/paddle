#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : jade
# @Date     : 2022/3/30 17:18
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import sys
sys.path.append("python_lib")
import paddle
from paddle import inference
from paddle.fluid import ir

if __name__ == '__main__':
    print(paddle.__version__)