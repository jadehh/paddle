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
"""
Dataset package.
"""

import paddle18.dataset.mnist
import paddle18.dataset.imikolov
import paddle18.dataset.imdb
import paddle18.dataset.cifar
import paddle18.dataset.movielens
import paddle18.dataset.conll05
import paddle18.dataset.uci_housing
import paddle18.dataset.sentiment
import paddle18.dataset.wmt14
import paddle18.dataset.wmt16
import paddle18.dataset.mq2007
import paddle18.dataset.flowers
import paddle18.dataset.voc2012
import paddle18.dataset.image

__all__ = [
    'mnist',
    'imikolov',
    'imdb',
    'cifar',
    'movielens',
    'conll05',
    'sentiment',
    'uci_housing',
    'wmt14',
    'wmt16',
    'mq2007',
    'flowers',
    'voc2012',
    'image',
]
