# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import logging
from ...common import get_logger
from .search_space_base import SearchSpaceBase
from .search_space_registry import SEARCHSPACE
from .base_layer import conv_bn_layer

__all__ = ["CombineSearchSpace"]

_logger = get_logger(__name__, level=logging.INFO)


class CombineSearchSpace(object):
    """
    Combine Search Space.
    Args:
        configs(list<tuple>): multi config.
    """

    def __init__(self, config_lists):
        self.lens = len(config_lists)
        self.spaces = []
        for config_list in config_lists:
            if isinstance(config_list, tuple):
                key, config = config_list
            elif isinstance(config_list, str):
                key = config_list
                config = None
            else:
                raise NotImplementedError(
                    'the type of config is Error!!! Please check the config information. Receive the type of config is {}'.
                    format(type(config_list)))
            self.spaces.append(self._get_single_search_space(key, config))
        self.init_tokens()

    def _get_single_search_space(self, key, config):
        """
        get specific model space based on key and config.

        Args:
            key(str): model space name.
            config(dict): basic config information.
        return:
            model space(class)
        """
        cls = SEARCHSPACE.get(key)
        assert cls != None, '{} is NOT a correct space, the space we support is {}'.format(
            key, SEARCHSPACE)

        if config is None:
            block_mask = None
            input_size = None
            output_size = None
            block_num = None
        else:
            if 'Block' not in cls.__name__:
                _logger.warn(
                    'if space is not a Block space, config is useless, current space is {}'.
                    format(cls.__name__))

            block_mask = config[
                'block_mask'] if 'block_mask' in config else None
            input_size = config[
                'input_size'] if 'input_size' in config else None
            output_size = config[
                'output_size'] if 'output_size' in config else None
            block_num = config['block_num'] if 'block_num' in config else None

        if 'Block' in cls.__name__:
            if block_mask == None and (block_num == None or
                                       input_size == None or
                                       output_size == None):
                raise NotImplementedError(
                    "block_mask or (block num and input_size and output_size) can NOT be None at the same time in Block SPACE!"
                )

        space = cls(input_size, output_size, block_num, block_mask=block_mask)
        return space

    def init_tokens(self, tokens=None):
        """
        Combine init tokens.
        """
        if tokens is None:
            tokens = []
            self.single_token_num = []
            for space in self.spaces:
                tokens.extend(space.init_tokens())
                self.single_token_num.append(len(space.init_tokens()))
            return tokens
        else:
            return tokens

    def range_table(self):
        """
        Combine range table.
        """
        range_tables = []
        for space in self.spaces:
            range_tables.extend(space.range_table())
        return range_tables

    def token2arch(self, tokens=None):
        """
        Combine model arch
        """
        if tokens is None:
            tokens = self.init_tokens()

        token_list = []
        start_idx = 0
        end_idx = 0

        for i in range(len(self.single_token_num)):
            end_idx += self.single_token_num[i]
            token_list.append(tokens[start_idx:end_idx])
            start_idx = end_idx

        model_archs = []
        for space, token in zip(self.spaces, token_list):
            model_archs.append(space.token2arch(token))

        return model_archs
