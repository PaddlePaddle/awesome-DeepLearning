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

import os
import socket
import logging
import numpy as np
import json
import hashlib
import time
import paddle.fluid as fluid
from ..common import SAController
from ..common import get_logger

from ..common import ControllerServer
from ..common import ControllerClient
from .search_space import SearchSpaceFactory

__all__ = ["SANAS"]

_logger = get_logger(__name__, level=logging.INFO)


class SANAS(object):
    """
    SANAS(Simulated Annealing Neural Architecture Search) is a neural architecture search algorithm 
    based on simulated annealing, used in discrete search task generally.

    Args:
        configs(list<tuple>): A list of search space configuration with format [(key, {input_size, 
                              output_size, block_num, block_mask})]. `key` is the name of search space 
                              with data type str. `input_size` and `output_size`  are input size and 
                              output size of searched sub-network. `block_num` is the number of blocks 
                              in searched network, `block_mask` is a list consists by 0 and 1, 0 means 
                              normal block, 1 means reduction block.
        server_addr(tuple): Server address, including ip and port of server. If ip is None or "", will 
                            use host ip if is_server = True. Default: ("", 8881).
        init_temperature(float): Initial temperature in SANAS. If init_temperature and init_tokens are None, 
                                 default initial temperature is 10.0, if init_temperature is None and 
                                 init_tokens is not None, default initial temperature is 1.0. The detail 
                                 configuration about the init_temperature please reference Note. Default: None.
        reduce_rate(float): Reduce rate in SANAS. The detail configuration about the reduce_rate please 
                            reference Note. Default: 0.85.
        search_steps(int): The steps of searching. Default: 300.
        init_tokens(list|None): Initial token. If init_tokens is None, SANAS will random generate initial 
                                tokens. Default: None.
        save_checkpoint(string|None): The directory of checkpoint to save, if set to None, not save checkpoint.
                                      Default: 'nas_checkpoint'.
        load_checkpoint(string|None): The directory of checkpoint to load, if set to None, not load checkpoint. 
                                      Default: None.
        is_server(bool): Whether current host is controller server. Default: True.

    .. note::
        - Why need to set initial temperature and reduce rate:

          - SA algorithm preserve a base token(initial token is the first base token, can be set by 
            yourself or random generate) and base score(initial score is -1), next token will be 
            generated based on base token. During the search, if the score which is obtained by the 
            model corresponding to the token is greater than the score which is saved in SA corresponding to 
            base token, current token saved as base token certainly; if score which is obtained by the model 
            corresponding to the token is less than the score which is saved in SA correspinding to base token, 
            current token saved as base token with a certain probability.
          - For initial temperature, higher is more unstable, it means that SA has a strong possibility to save 
            current token as base token if current score is smaller than base score saved in SA.
          - For initial temperature, lower is more stable, it means that SA has a small possibility to save 
            current token as base token if current score is smaller than base score saved in SA.
          - For reduce rate, higher means SA algorithm has slower convergence.
          - For reduce rate, lower means SA algorithm has faster convergence.

        - How to set initial temperature and reduce rate:

          - If there is a better initial token, and want to search based on this token, we suggest start search 
            experiment in the steady state of the SA algorithm, initial temperature can be set to a small value, 
            such as 1.0, and reduce rate can be set to a large value, such as 0.85. If you want to start search 
            experiment based on the better token with greedy algorithm, which only saved current token as base 
            token if current score higher than base score saved in SA algorithm, reduce rate can be set to a 
            extremely small value, such as 0.85 ** 10.

          - If initial token is generated randomly, it means initial token is a worse token, we suggest start 
            search experiment in the unstable state of the SA algorithm, explore all random tokens as much as 
            possible, and get a better token. Initial temperature can be set a higher value, such as 1000.0, 
            and reduce rate can be set to a small value.
    """

    def __init__(self,
                 configs,
                 server_addr=("", 8881),
                 init_temperature=None,
                 reduce_rate=0.85,
                 search_steps=300,
                 init_tokens=None,
                 save_checkpoint='nas_checkpoint',
                 load_checkpoint=None,
                 is_server=True):
        if not is_server:
            assert server_addr[
                0] != "", "You should set the IP and port of server when is_server is False."
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._is_server = is_server
        self._configs = configs
        self._init_tokens = init_tokens
        self._client_name = hashlib.md5(
            str(time.time() + np.random.randint(1, 10000)).encode(
                "utf-8")).hexdigest()
        self._key = str(self._configs)
        self._current_tokens = init_tokens

        self._server_ip, self._server_port = server_addr
        if self._server_ip == None or self._server_ip == "":
            self._server_ip = self._get_host_ip()

        factory = SearchSpaceFactory()
        self._search_space = factory.get_search_space(configs)

        # create controller server
        if self._is_server:
            init_tokens = self._search_space.init_tokens(self._init_tokens)
            range_table = self._search_space.range_table()
            range_table = (len(range_table) * [0], range_table)
            _logger.info("range table: {}".format(range_table))

            if load_checkpoint != None:
                assert os.path.exists(
                    load_checkpoint
                ) == True, 'load checkpoint file NOT EXIST!!! Please check the directory of checkpoint!!!'
                checkpoint_path = os.path.join(load_checkpoint,
                                               'sanas.checkpoints')
                with open(checkpoint_path, 'r') as f:
                    scene = json.load(f)
                preinit_tokens = scene['_tokens']
                prereward = scene['_reward']
                premax_reward = scene['_max_reward']
                prebest_tokens = scene['_best_tokens']
                preiter = scene['_iter']
                psearched = scene['_searched']
            else:
                preinit_tokens = init_tokens
                prereward = -1
                premax_reward = -1
                prebest_tokens = None
                preiter = 0
                psearched = None

            self._controller = SAController(
                range_table,
                self._reduce_rate,
                self._init_temperature,
                max_try_times=50000,
                init_tokens=preinit_tokens,
                reward=prereward,
                max_reward=premax_reward,
                iters=preiter,
                best_tokens=prebest_tokens,
                constrain_func=None,
                checkpoints=save_checkpoint,
                searched=psearched)

            max_client_num = 100
            self._controller_server = ControllerServer(
                controller=self._controller,
                address=(self._server_ip, self._server_port),
                max_client_num=max_client_num,
                search_steps=search_steps,
                key=self._key)
            self._controller_server.start()
            server_port = self._controller_server.port()

        self._controller_client = ControllerClient(
            self._server_ip,
            self._server_port,
            key=self._key,
            client_name=self._client_name)

        if is_server and load_checkpoint != None:
            self._iter = scene['_iter']
        else:
            self._iter = 0

    def _get_host_ip(self):
        try:
            return socket.gethostbyname(socket.gethostname())
        except:
            return socket.gethostbyname('localhost')

    def tokens2arch(self, tokens):
        """
        Convert tokens to model architectures.
        Args
            tokens<list>: A list of token. The length and range based on search space.:
        Returns:
            list<function>: A model architecture instance according to tokens.
        """
        return self._search_space.token2arch(tokens)

    def current_info(self):
        """
        Get current information, including best tokens, best reward in all the search, and current token.
        Returns:
            dict<name, value>: a dictionary include best tokens, best reward and current reward.
        """
        current_dict = self._controller_client.request_current_info()
        return current_dict

    def next_archs(self):
        """
        Get next model architectures.
        Returns:
            list<function>: A list of instance of model architecture.
        """
        self._current_tokens = self._controller_client.next_tokens()
        _logger.info("current tokens: {}".format(self._current_tokens))
        archs = self._search_space.token2arch(self._current_tokens)
        return archs

    def reward(self, score):
        """
        Return reward of current searched network.
        Args:
            score(float): The score of current searched network, bigger is better.
        Returns:
            bool: True means updating successfully while false means failure.
        """
        self._iter += 1
        return self._controller_client.update(self._current_tokens, score,
                                              self._iter)
