#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : __init__.py.py
# @Author   : jade
# @Date     : 2021/4/30 15:35
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
# Copyright (c) 2016 paddle18paddle18 Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from paddle18.check_import_scipy import check_import_scipy

check_import_scipy(os.name)

try:
    from paddle18.version import full_version as __version__
    from paddle18.version import commit as __git_commit__

except ImportError:
    import sys
    sys.stderr.write('''Warning with import paddle18: you should not
     import paddle18 from the source directory; please install paddle18paddle18*.whl firstly.'''
                     )

import paddle18.reader
import paddle18.dataset
import paddle18.batch
import paddle18.compat
import paddle18.distributed
batch = batch.batch
import paddle18.sysconfig
