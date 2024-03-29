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
    name="paddle-gpu-clean",
    version="2.1.0",
    keywords=("pip", "paddle", ""),
    description="paddle",
    long_description="xxx",
    license="MIT Licence",
    classifiers=[
       "Operating System :: POSIX :: Linux",  # 编程语言
    ],

    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["decorator","six==1.16.0","numpy","protobuf","requests","pillow","gast","astor"]  # 这个项目需要的第三方库
)
