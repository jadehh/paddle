#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : jade
# @Date     : 2022/12/6 16:38
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import sys
sys.path.append(r"H:\Libs\paddle\v2.3.2-gpu-win-py3.6\python_libs")
import paddle
from paddle import inference
from paddle.fluid import ir
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
from paddle.fluid.ir import *
from paddle.fluid.proto.pass_desc_pb2 import *
if __name__ == '__main__':
    print(paddle.__version__)