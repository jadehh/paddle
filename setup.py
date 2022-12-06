#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py
# @Author   : jade
# @Date     : 2022/12/6 18:53
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : jade
# @Date     : 2022/12/6 16:40
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages
import os
with open("README.md","r") as fh:
    long_description = fh.read()
def find_packages(path,pack_list):
    for file_path in os.listdir(path):
        if os.path.isdir(os.path.join(path,file_path)):
            if "__pycache__" != file_path:
                new_path = path.replace("/", ".")
                if new_path:
                    pack_list.append("{}.{}".format(new_path,file_path))
                else:
                    pack_list.append("{}.{}".format(path, file_path))
                find_packages(os.path.join(path,file_path),pack_list)


if __name__ == '__main__':
    pack_list = ["paddle"]
    find_packages("paddle",pack_list)
setup(
    name="paddle",
    version="2.1.0",
    keywords=("pip", "paddle", ""),
    description="paddle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT Licence",
    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",
    packages=pack_list,
    package_data={'': ["*.md"]},
    include_package_data=True,
    platforms="any",
    install_requires=["decorator==5.1.1","six==1.16.0","protobuf==3.19.4","requests==2.27.1","astor==0.8.1","opt_einsum==3.3.0","gast==0.5.3"] # 这个项目需要的第三方库
)