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
    platforms="win",
    install_requires=["numpy","astor","pillow","scipy","requests","protobuf","six","decorator","shapely==1.8.0","pyclipper"]  # 这个项目需要的第三方库
)
