#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : jade
# @Date     : 2022/3/30 17:19
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages

setup(
    name="paddle",
    version="2.0.2",
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
    install_requires=["decorator==5.1.1","six==1.16.0","numpy==1.19.5","protobuf==3.20.1","requests==2.27.1",
                      "pillow==8.4.0","astor==0.8.1","opt_einsum==3.3.0","shapely==1.8.0","pyclipper==1.3.0",
                      "gast==0.5.3"]  # 这个项目需要的第三方库
)
