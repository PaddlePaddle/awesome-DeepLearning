#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from .utils.utils import get_paddle_version
pd_ver = get_paddle_version()
import paddle
if pd_ver == 185:
    Layer = paddle.fluid.dygraph.Layer
else:
    Layer = paddle.nn.Layer

_cnt = 0


def counter():
    global _cnt
    _cnt += 1
    return _cnt


class BaseBlock(Layer):
    def __init__(self, key=None):
        super(BaseBlock, self).__init__()
        if key is not None:
            self._key = str(key)
        else:
            self._key = self.__class__.__name__ + str(counter())

    # set SuperNet class
    def set_supernet(self, supernet):
        self.__dict__['supernet'] = supernet

    @property
    def key(self):
        return self._key


class Block(BaseBlock):
    """
    Model is composed of nest blocks.

    Parameters:
        fn(paddle.nn.Layer): instance of super layers, such as: SuperConv2D(3, 5, 3).
        fixed(bool, optional): whether to fix the shape of the weight in this layer. Default: False.
        key(str, optional): key of this layer, one-to-one correspondence between key and candidate config. Default: None.
    """

    def __init__(self, fn, fixed=False, key=None):
        super(Block, self).__init__(key)
        self.fn = fn
        self.fixed = fixed
        self.candidate_config = self.fn.candidate_config

    def forward(self, *inputs, **kwargs):
        out = self.supernet.layers_forward(self, *inputs, **kwargs)
        return out
