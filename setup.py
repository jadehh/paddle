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
    version="2.2",
    keywords=("pip", "paddle", ""),
    description="paddle",
    long_description="xxx",
    license="MIT Licence",

    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="linux",
    install_requires=["decorator==5.1.1","six==1.16.0","numpy==1.19.5","protobuf==3.19.4","requests==2.27.1","pillow==8.4.0","astor==0.8.1","scipy==1.5.4"]  # 这个项目需要的第三方库
)
