# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import socket
import time
from .log_helper import get_logger
from threading import Thread
from .lock import lock, unlock

__all__ = ['ControllerServer']

_logger = get_logger(__name__, level=logging.INFO)


class ControllerServer(object):
    """The controller wrapper with a socket server to handle the request of search agent.
    Args:
        controller(slim.searcher.Controller): The controller used to generate tokens.
        address(tuple): The address of current server binding with format (ip, port). Default: ('', 0).
                        which means setting ip automatically
        max_client_num(int): The maximum number of clients connecting to current server simultaneously. Default: 100.
        search_steps(int|None): The total steps of searching. None means never stopping. Default: None 
        key(str|None): Config information. Default: None.
    """

    def __init__(self,
                 controller=None,
                 address=('', 0),
                 max_client_num=100,
                 search_steps=None,
                 key=None):
        """
        """
        self._controller = controller
        self._address = address
        self._max_client_num = max_client_num
        self._search_steps = search_steps
        self._closed = False
        self._port = address[1]
        self._ip = address[0]
        self._key = key
        self._client_num = 0
        self._client = dict()
        self._compare_time = 172800  ### 48 hours

    def start(self):
        self._socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_server.bind(self._address)
        self._socket_server.listen(self._max_client_num)
        self._port = self._socket_server.getsockname()[1]
        self._ip = self._socket_server.getsockname()[0]
        _logger.info("ControllerServer Start!!!")
        _logger.debug("ControllerServer - listen on: [{}:{}]".format(
            self._ip, self._port))
        thread = Thread(target=self.run)
        thread.setDaemon(True)
        thread.start()
        return str(thread)

    def close(self):
        """Close the server."""
        self._closed = True
        _logger.info("server closed!")

    def port(self):
        """Get the port."""
        return self._port

    def ip(self):
        """Get the ip."""
        return self._ip

    def run(self):
        """Start the server.
        """
        _logger.info("Controller Server run...")
        try:
            while ((self._search_steps is None) or
                   (self._controller._iter <
                    (self._search_steps))) and not self._closed:
                conn, addr = self._socket_server.accept()
                message = conn.recv(1024).decode()
                _logger.debug(message)
                if message.strip("\n") == "next_tokens":
                    tokens = self._controller.next_tokens()
                    tokens = ",".join([str(token) for token in tokens])
                    conn.send(tokens.encode())
                elif message.strip("\n") == "current_info":
                    current_info = dict()
                    current_info['best_tokens'] = self._controller.best_tokens
                    current_info['best_reward'] = self._controller.max_reward
                    current_info[
                        'current_tokens'] = self._controller.current_tokens
                    conn.send(str(current_info).encode())
                else:
                    _logger.debug("recv message from {}: [{}]".format(addr,
                                                                      message))
                    messages = message.strip('\n').split("\t")
                    if (len(messages) < 5) or (messages[0] != self._key):
                        _logger.debug("recv noise from {}: [{}]".format(
                            addr, message))
                        continue
                    tokens = messages[1]
                    reward = messages[2]
                    iter = messages[3]
                    client_name = messages[4]

                    one_step_time = -1
                    if client_name in self._client.keys():
                        current_time = time.time() - self._client[client_name]
                        if current_time > one_step_time:
                            one_step_time = current_time
                            self._compare_time = 2 * one_step_time

                    if client_name not in self._client.keys():
                        self._client[client_name] = time.time()
                        self._client_num += 1

                    self._client[client_name] = time.time()

                    for key_client in self._client.keys():
                        ### if a client not request token in double train one tokens' time, we think this client was stoped.
                        if (time.time() - self._client[key_client]
                            ) > self._compare_time and len(self._client.keys(
                            )) > 1:
                            self._client.pop(key_client)
                            self._client_num -= 1
                    _logger.debug(
                        "client: {}, client_num: {}, compare_time: {}".format(
                            self._client, self._client_num,
                            self._compare_time))
                    tokens = [int(token) for token in tokens.split(",")]
                    self._controller.update(tokens,
                                            float(reward),
                                            int(iter), int(self._client_num))
                    response = "ok"
                    conn.send(response.encode())
                    _logger.debug("send message to {}: [{}]".format(addr,
                                                                    tokens))
                conn.close()
        except Exception as err:
            _logger.error(err)
        finally:
            self._socket_server.close()
            self.close()
