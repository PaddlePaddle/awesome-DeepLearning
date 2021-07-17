# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import signal
import zmq
import socket
import logging
import time
import threading
import six
if six.PY2:
    import cPickle as pickle
else:
    import pickle
from .log_helper import get_logger
from .rl_controller.utils import compute_grad, ConnectMessage

_logger = get_logger(__name__, level=logging.INFO)


class Client(object):
    def __init__(self, controller, address, client_name):
        self._controller = controller
        self._address = address
        self._ip = self._address[0]
        self._port = self._address[1]
        self._client_name = client_name
        self._params_dict = None
        self.init_wait = False
        self._connect_server()

    def _connect_server(self):
        self._ctx = zmq.Context()
        self._client_socket = self._ctx.socket(zmq.REQ)
        ### NOTE: change the method to exit client when server is dead if there are better solutions
        self._client_socket.setsockopt(zmq.RCVTIMEO,
                                       ConnectMessage.TIMEOUT * 1000)
        client_address = "{}:{}".format(self._ip, self._port)
        self._client_socket.connect("tcp://{}".format(client_address))
        self._client_socket.send_multipart([
            pickle.dumps(ConnectMessage.INIT), pickle.dumps(self._client_name)
        ])
        message = self._client_socket.recv_multipart()
        if pickle.loads(message[0]) != ConnectMessage.INIT_DONE:
            _logger.error("Client {} init failure, Please start it again".
                          format(self._client_name))
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)
        _logger.info("Client {}: connect to server success!!!".format(
            self._client_name))
        _logger.debug("Client {}: connect to server {}".format(
            self._client_name, client_address))

    def _connect_wait_socket(self, port):
        self._wait_socket = self._ctx.socket(zmq.REQ)
        wait_address = "{}:{}".format(self._ip, port)
        self._wait_socket.connect("tcp://{}".format(wait_address))
        self._wait_socket.send_multipart([
            pickle.dumps(ConnectMessage.WAIT_PARAMS),
            pickle.dumps(self._client_name)
        ])
        message = self._wait_socket.recv_multipart()
        return pickle.loads(message[0])

    def next_tokens(self, obs, is_inference=False):
        _logger.debug("Client: requests for weight {}".format(
            self._client_name))
        self._client_socket.send_multipart([
            pickle.dumps(ConnectMessage.GET_WEIGHT),
            pickle.dumps(self._client_name)
        ])
        try:
            message = self._client_socket.recv_multipart()
        except zmq.error.Again as e:
            _logger.error(
                "CANNOT recv params from server in next_archs, Please check whether the server is alive!!! {}".
                format(e))
            os._exit(0)
        self._params_dict = pickle.loads(message[0])
        tokens = self._controller.next_tokens(
            obs, params_dict=self._params_dict, is_inference=is_inference)
        _logger.debug("Client: client_name is {}, current token is {}".format(
            self._client_name, tokens))
        return tokens

    def update(self, rewards, **kwargs):
        assert self._params_dict != None, "Please call next_token to get token first, then call update"
        current_params_dict = self._controller.update(
            rewards, self._params_dict, **kwargs)
        params_grad = compute_grad(current_params_dict, self._params_dict)
        _logger.debug("Client: update weight {}".format(self._client_name))
        self._client_socket.send_multipart([
            pickle.dumps(ConnectMessage.UPDATE_WEIGHT),
            pickle.dumps(self._client_name), pickle.dumps(params_grad)
        ])
        _logger.debug("Client: update done {}".format(self._client_name))

        try:
            message = self._client_socket.recv_multipart()
        except zmq.error.Again as e:
            _logger.error(
                "CANNOT recv params from server in rewards, Please check whether the server is alive!!! {}".
                format(e))
            os._exit(0)

        if pickle.loads(message[0]) == ConnectMessage.WAIT:
            _logger.debug("Client: self.init_wait: {}".format(self.init_wait))
            if not self.init_wait:
                wait_port = pickle.loads(message[1])
                wait_signal = self._connect_wait_socket(wait_port)
                self.init_wait = True
            else:
                wait_signal = pickle.loads(message[0])
            while wait_signal != ConnectMessage.OK:
                time.sleep(1)
                self._wait_socket.send_multipart([
                    pickle.dumps(ConnectMessage.WAIT_PARAMS),
                    pickle.dumps(self._client_name)
                ])
                wait_signal = self._wait_socket.recv_multipart()
                wait_signal = pickle.loads(wait_signal[0])
                _logger.debug("Client: {} {}".format(self._client_name,
                                                     wait_signal))

        return pickle.loads(message[0])

    def __del__(self):
        try:
            self._client_socket.send_multipart([
                pickle.dumps(ConnectMessage.EXIT),
                pickle.dumps(self._client_name)
            ])
            _ = self._client_socket.recv_multipart()
        except:
            pass
        self._client_socket.close()
