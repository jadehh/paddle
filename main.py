#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : jade
# @Date     : 2022/12/6 18:52
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import sys
sys.path.append(r"H:\Libs\paddle\v2.1.0-gpu-win-py3.6")
import paddle
from paddle import inference
if __name__ == '__main__':
    print(paddle.__version__)