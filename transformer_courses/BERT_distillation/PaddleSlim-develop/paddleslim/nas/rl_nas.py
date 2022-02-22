# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import os
import socket
import logging
import numpy as np
import json
import hashlib
import time
import paddle.fluid as fluid
from ..common.rl_controller.utils import RLCONTROLLER
from ..common import get_logger

from ..common import Server
from ..common import Client
from .search_space import SearchSpaceFactory

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['RLNAS']


class RLNAS(object):
    """ 
    Controller with Reinforcement Learning. 
    Args:
        key(str): The actual reinforcement learning method. Current support in paddleslim is `LSTM` and `DDPG`.
        configs(list<tuple>): A list of search space configuration with format [(key, {input_size,
                              output_size, block_num, block_mask})]. `key` is the name of search space
                              with data type str. `input_size` and `output_size`  are input size and
                              output size of searched sub-network. `block_num` is the number of blocks
                              in searched network, `block_mask` is a list consists by 0 and 1, 0 means
                              normal block, 1 means reduction block.
        use_gpu(bool): Whether to use gpu in controller. Default: False.
        server_addr(tuple): Server address, including ip and port of server. If ip is None or "", will
                            use host ip if is_server = True. Default: ("", 8881).
        is_server(bool): Whether current host is controller server. Default: True.
        is_sync(bool): Whether to update controller in synchronous mode. Default: False.
        save_controller(str|None): The directory of controller to save, if set to None, not save checkpoint.
                                      Default: None.
        load_controller(str|None): The directory of controller to load, if set to None, not load checkpoint.
                                      Default: None.
        **kwargs: Additional keyword arguments. 
    """

    def __init__(self,
                 key,
                 configs,
                 use_gpu=False,
                 server_addr=("", 8881),
                 is_server=True,
                 is_sync=False,
                 save_controller=None,
                 load_controller=None,
                 **kwargs):
        if not is_server:
            assert server_addr[
                0] != "", "You should set the IP and port of server when is_server is False."

        self._configs = configs
        factory = SearchSpaceFactory()
        self._search_space = factory.get_search_space(configs)
        self.range_tables = self._search_space.range_table()
        self.save_controller = save_controller
        self.load_controller = load_controller
        self._is_server = is_server

        if key.upper() in ['DDPG']:
            try:
                import parl
            except ImportError as e:
                _logger.error(
                    "If you want to use DDPG in RLNAS, please pip install parl first. Now states: {}".
                    format(e))
                os._exit(1)

        cls = RLCONTROLLER.get(key.upper())

        server_ip, server_port = server_addr
        if server_ip == None or server_ip == "":
            server_ip = self._get_host_ip()

        self._controller = cls(range_tables=self.range_tables,
                               use_gpu=use_gpu,
                               **kwargs)

        if is_server:
            max_client_num = 300
            self._controller_server = Server(
                controller=self._controller,
                address=(server_ip, server_port),
                is_sync=is_sync,
                save_controller=self.save_controller,
                load_controller=self.load_controller)
            self._controller_server.start()

        self._client_name = hashlib.md5(
            str(time.time() + np.random.randint(1, 10000)).encode(
                "utf-8")).hexdigest()
        self._controller_client = Client(
            controller=self._controller,
            address=(server_ip, server_port),
            client_name=self._client_name)

        self._current_tokens = None

    def _get_host_ip(self):
        try:
            return socket.gethostbyname(socket.gethostname())
        except:
            return socket.gethostbyname('localhost')

    def next_archs(self, obs=None):
        """ 
        Get next archs
        Args:
            obs(int|np.array): observations in env.
        """
        archs = []
        self._current_tokens = self._controller_client.next_tokens(obs)
        _logger.info("current tokens: {}".format(self._current_tokens))
        for token in self._current_tokens:
            archs.append(self._search_space.token2arch(token))

        return archs

    @property
    def tokens(self):
        return self._current_tokens

    def reward(self, rewards, **kwargs):
        """ 
        reward the score and to train controller
        Args:
            rewards(float|list<float>): rewards get by tokens.
            **kwargs: Additional keyword arguments. 
        """
        return self._controller_client.update(rewards, **kwargs)

    def final_archs(self, batch_obs):
        """
        Get finally architecture
        Args:
            batch_obs(int|np.array): observations in env.
        """
        final_tokens = self._controller_client.next_tokens(
            batch_obs, is_inference=True)
        self._current_tokens = final_tokens
        _logger.info("Final tokens: {}".format(final_tokens))
        archs = []
        for token in final_tokens:
            arch = self._search_space.token2arch(token)
            archs.append(arch)

        return archs

    def tokens2arch(self, tokens):
        """
        Convert tokens to model architectures.
        Args
            tokens<list>: A list of token. The length and range based on search space.:
        Returns:
            list<function>: A model architecture instance according to tokens.
        """
        return self._search_space.token2arch(tokens)
