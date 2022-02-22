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

import zmq
import socket
import signal
import six
import os
if six.PY2:
    import cPickle as pickle
else:
    import pickle
import logging
import time
import threading
from .log_helper import get_logger
from .rl_controller.utils import add_grad, ConnectMessage

_logger = get_logger(__name__, level=logging.INFO)


class Server(object):
    def __init__(self,
                 controller,
                 address,
                 is_sync=False,
                 load_controller=None,
                 save_controller=None):
        self._controller = controller
        self._address = address
        self._ip = self._address[0]
        self._port = self._address[1]
        self._is_sync = is_sync
        self._done = False
        self._load_controller = load_controller
        self._save_controller = save_controller
        ### key-value : client_name-update_times
        self._client_dict = dict()
        self._client = list()
        self._lock = threading.Lock()
        self._server_alive = True
        self._max_update_times = 0

    def close(self):
        self._server_alive = False
        _logger.info("server closed")
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)

    def start(self):
        self._ctx = zmq.Context()
        ### main socket
        self._server_socket = self._ctx.socket(zmq.REP)
        server_address = "{}:{}".format(self._ip, self._port)
        self._server_socket.bind("tcp://{}".format(server_address))
        self._server_socket.linger = 0
        _logger.info("ControllerServer Start!!!")
        _logger.debug("ControllerServer - listen on: [{}]".format(
            server_address))
        thread = threading.Thread(target=self.run, args=())
        thread.setDaemon(True)
        thread.start()

        if self._load_controller:
            assert os.path.exists(
                self._load_controller
            ), "controller checkpoint is not exist, please check your directory: {}".format(
                self._load_controller)

            with open(
                    os.path.join(self._load_controller, 'rlnas.params'),
                    'rb') as f:
                self._params_dict = pickle.load(f)
            _logger.info("Load params done")

        else:
            self._params_dict = self._controller.param_dict

        if self._is_sync:
            self._wait_socket = self._ctx.socket(zmq.REP)
            self._wait_port = self._wait_socket.bind_to_random_port(
                addr="tcp://*")
            self._wait_socket_linger = 0
            wait_thread = threading.Thread(
                target=self._wait_for_params, args=())
            wait_thread.setDaemon(True)
            wait_thread.start()

    def _wait_for_params(self):
        try:
            while self._server_alive:
                message = self._wait_socket.recv_multipart()
                cmd = pickle.loads(message[0])
                client_name = pickle.loads(message[1])
                if cmd == ConnectMessage.WAIT_PARAMS:
                    _logger.debug("Server: wait for params")
                    self._lock.acquire()
                    self._wait_socket.send_multipart([
                        pickle.dumps(ConnectMessage.OK)
                        if self._done else pickle.dumps(ConnectMessage.WAIT)
                    ])
                    if self._done and client_name in self._client:
                        self._client.remove(client_name)
                    if len(self._client) == 0:
                        if self._save_controller != False:
                            self.save_params()
                        self._done = False
                    self._lock.release()
                else:
                    _logger.error("Error message {}".format(message))
                    raise NotImplementedError
        except Exception as err:
            logger.error(err)

    def run(self):
        try:
            while self._server_alive:
                try:
                    sum_params_dict = dict()
                    message = self._server_socket.recv_multipart()
                    cmd = pickle.loads(message[0])
                    client_name = pickle.loads(message[1])
                    if cmd == ConnectMessage.INIT:
                        self._server_socket.send_multipart(
                            [pickle.dumps(ConnectMessage.INIT_DONE)])
                        _logger.debug("Server: init client {}".format(
                            client_name))
                        self._client_dict[client_name] = 0
                    elif cmd == ConnectMessage.GET_WEIGHT:
                        self._lock.acquire()
                        _logger.debug("Server: get weight {}".format(
                            client_name))
                        self._server_socket.send_multipart(
                            [pickle.dumps(self._params_dict)])
                        _logger.debug("Server: send params done {}".format(
                            client_name))
                        self._lock.release()
                    elif cmd == ConnectMessage.UPDATE_WEIGHT:
                        _logger.info("Server: update {}".format(client_name))
                        params_dict_grad = pickle.loads(message[2])
                        if self._is_sync:
                            if not sum_params_dict:
                                sum_params_dict = self._params_dict
                            self._lock.acquire()
                            sum_params_dict = add_grad(sum_params_dict,
                                                       params_dict_grad)
                            self._client.append(client_name)
                            self._lock.release()

                            if len(self._client) == len(
                                    self._client_dict.items()):
                                self._done = True
                                self._params_dict = sum_params_dict
                                del sum_params_dict

                            self._server_socket.send_multipart([
                                pickle.dumps(ConnectMessage.WAIT),
                                pickle.dumps(self._wait_port)
                            ])
                        else:
                            self._lock.acquire()
                            self._params_dict = add_grad(self._params_dict,
                                                         params_dict_grad)
                            self._client_dict[client_name] += 1
                            if self._client_dict[
                                    client_name] > self._max_update_times:
                                self._max_update_times = self._client_dict[
                                    client_name]
                            self._lock.release()
                            if self._save_controller != False:
                                self.save_params()
                            self._server_socket.send_multipart(
                                [pickle.dumps(ConnectMessage.OK)])

                    elif cmd == ConnectMessage.EXIT:
                        self._client_dict.pop(client_name)
                        if client_name in self._client:
                            self._client.remove(client_name)
                        self._server_socket.send_multipart(
                            [pickle.dumps(ConnectMessage.EXIT)])
                except zmq.error.Again as e:
                    _logger.error(e)
            self.close()

        except Exception as err:
            _logger.error(err)
        finally:
            self._server_socket.close(0)
            if self._is_sync:
                self._wait_socket.close(0)
            self.close()

    def save_params(self):
        if self._save_controller:
            if not os.path.exists(self._save_controller):
                os.makedirs(self._save_controller)
            output_dir = self._save_controller
        else:
            if not os.path.exists('./.rlnas_controller'):
                os.makedirs('./.rlnas_controller')
            output_dir = './.rlnas_controller'

        with open(os.path.join(output_dir, 'rlnas.params'), 'wb') as f:
            pickle.dump(self._params_dict, f)
        _logger.debug("Save params done")
