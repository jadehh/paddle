#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : jade
# @Date     : 2021/4/30 14:05
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages

setup(
    name="paddle",
    version="1.7.2",
    keywords=("pip", "paddle", ""),
    description="paddle",
    long_description="xxx",
    license="MIT Licence",

    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["numpy","prettytable","pyyaml","funcsigs","pillow","scipy","rarfile","nltk","requests","objgraph","protobuf","six"]  # 这个项目需要的第三方库
)
