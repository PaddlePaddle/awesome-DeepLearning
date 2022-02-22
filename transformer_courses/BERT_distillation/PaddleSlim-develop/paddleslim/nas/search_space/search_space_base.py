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

import logging
from ...common import get_logger

__all__ = ['SearchSpaceBase']

_logger = get_logger(__name__, level=logging.INFO)


class SearchSpaceBase(object):
    """Controller for Neural Architecture Search.
    """

    def __init__(self, input_size, output_size, block_num, block_mask, *args):
        """init model config
        """
        self.input_size = input_size
        self.output_size = output_size
        self.block_num = block_num
        self.block_mask = block_mask
        if self.block_mask != None:
            assert isinstance(self.block_mask,
                              list), 'Block_mask must be a list.'
            _logger.warn(
                "If block_mask is NOT None, we will use block_mask as major configs!"
            )
            self.block_num = None

    def init_tokens(self):
        """Get init tokens in search space.
        """
        raise NotImplementedError('Abstract method.')

    def range_table(self):
        """Get range table of current search space.
        """
        raise NotImplementedError('Abstract method.')

    def token2arch(self, tokens):
        """Create networks for training and evaluation according to tokens.
        Args:
            tokens(list<int>): The tokens which represent a network.
        Return:
            model arch 
        """
        raise NotImplementedError('Abstract method.')

    def super_net(self):
        """This function is just used in one shot NAS strategy. Return a super graph."""
        raise NotImplementedError('Abstract method.')
