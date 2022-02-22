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

import math
import logging
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.layers import RNNCell, LSTMCell, rnn
from paddle.fluid.contrib.layers import basic_lstm
from ...controller import RLBaseController
from ...log_helper import get_logger
from ..utils import RLCONTROLLER

_logger = get_logger(__name__, level=logging.INFO)

uniform_initializer = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x)


class lstm_cell(RNNCell):
    def __init__(self, num_layers, hidden_size):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells = []

        param_attr = ParamAttr(initializer=uniform_initializer(
            1.0 / math.sqrt(hidden_size)))
        bias_attr = ParamAttr(initializer=uniform_initializer(
            1.0 / math.sqrt(hidden_size)))
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, param_attr, bias_attr))

    def call(self, inputs, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](inputs, states[i])
            new_states.append(new_state)
        return out, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


@RLCONTROLLER.register
class LSTM(RLBaseController):
    def __init__(self, range_tables, use_gpu=False, **kwargs):
        self.use_gpu = use_gpu
        self.range_tables = range_tables
        self.lstm_num_layers = kwargs.get('lstm_num_layers') or 1
        self.hidden_size = kwargs.get('hidden_size') or 100
        self.temperature = kwargs.get('temperature') or None
        self.controller_lr = kwargs.get('controller_lr') or 1e-4
        self.decay_steps = kwargs.get('controller_decay_steps') or None
        self.decay_rate = kwargs.get('controller_decay_rate') or None
        self.tanh_constant = kwargs.get('tanh_constant') or None
        self.decay = kwargs.get('decay') or 0.99
        self.weight_entropy = kwargs.get('weight_entropy') or None
        self.controller_batch_size = kwargs.get('controller_batch_size') or 1

        self.max_range_table = max(self.range_tables) + 1

        self._create_parameter()
        self._build_program()

        self.place = paddle.CUDAPlace(0) if self.use_gpu else paddle.CPUPlace()
        self.exe = paddle.static.Executor(self.place)
        self.exe.run(paddle.static.default_startup_program())

        self.param_dict = self.get_params(self.learn_program)

    def _lstm(self, inputs, hidden, cell, token_idx):
        cells = lstm_cell(self.lstm_num_layers, self.hidden_size)
        output, new_states = cells.call(inputs, states=([[hidden, cell]]))
        logits = paddle.static.nn.fc(new_states[0],
                                     self.range_tables[token_idx])

        if self.temperature is not None:
            logits = logits / self.temperature
        if self.tanh_constant is not None:
            logits = self.tanh_constant * paddle.tanh(logits)

        return logits, output, new_states

    def _create_parameter(self):
        self.g_emb = paddle.static.create_parameter(
            name='emb_g',
            shape=(self.controller_batch_size, self.hidden_size),
            dtype='float32',
            default_initializer=uniform_initializer(1.0))
        self.baseline = fluid.layers.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name='baseline')
        self.baseline.stop_gradient = True

    def _network(self, hidden, cell, init_actions=None, is_inference=False):
        actions = []
        entropies = []
        sample_log_probs = []

        with fluid.unique_name.guard('Controller'):
            self._create_parameter()
            inputs = self.g_emb

            for idx in range(len(self.range_tables)):
                logits, output, states = self._lstm(
                    inputs, hidden, cell, token_idx=idx)
                hidden, cell = np.squeeze(states)
                probs = paddle.nn.functional.softmax(logits, axis=1)
                if is_inference:
                    action = paddle.argmax(probs, axis=1)
                else:
                    if init_actions:
                        action = paddle.slice(
                            init_actions,
                            axes=[1],
                            starts=[idx],
                            ends=[idx + 1])
                        action = paddle.squeeze(action, axis=[1])
                        action.stop_gradient = True
                    else:
                        action = fluid.layers.sampling_id(probs)
                actions.append(action)
                log_prob = paddle.nn.functional.softmax_with_cross_entropy(
                    logits,
                    paddle.reshape(
                        action, shape=[paddle.shape(action), 1]),
                    axis=1)
                sample_log_probs.append(log_prob)

                entropy = log_prob * paddle.exp(-1 * log_prob)
                entropy.stop_gradient = True
                entropies.append(entropy)

                action_emb = paddle.cast(action, dtype=np.int64)
                inputs = paddle.static.nn.embedding(
                    action_emb,
                    size=(self.max_range_table, self.hidden_size),
                    param_attr=paddle.ParamAttr(
                        name='emb_w', initializer=uniform_initializer(1.0)))

            self.sample_log_probs = paddle.concat(sample_log_probs, axis=0)

            entropies = paddle.stack(entropies)
            self.sample_entropies = paddle.sum(entropies)

        return actions

    def _build_program(self, is_inference=False):
        self.pred_program = paddle.static.Program()
        self.learn_program = paddle.static.Program()
        with paddle.static.program_guard(self.pred_program):
            self.g_emb = paddle.static.create_parameter(
                name='emb_g',
                shape=(self.controller_batch_size, self.hidden_size),
                dtype='float32',
                default_initializer=uniform_initializer(1.0))

            paddle.assign(
                fluid.layers.uniform_random(shape=self.g_emb.shape), self.g_emb)
            hidden = fluid.data(name='hidden', shape=[None, self.hidden_size])
            cell = fluid.data(name='cell', shape=[None, self.hidden_size])
            self.tokens = self._network(hidden, cell, is_inference=is_inference)

        with paddle.static.program_guard(self.learn_program):
            hidden = fluid.data(name='hidden', shape=[None, self.hidden_size])
            cell = fluid.data(name='cell', shape=[None, self.hidden_size])
            init_actions = fluid.data(
                name='init_actions',
                shape=[None, len(self.range_tables)],
                dtype='int64')
            self._network(hidden, cell, init_actions=init_actions)

            rewards = fluid.data(name='rewards', shape=[None])
            self.rewards = paddle.mean(rewards)

            if self.weight_entropy is not None:
                self.rewards += self.weight_entropy * self.sample_entropies

            self.sample_log_probs = paddle.sum(self.sample_log_probs)

            paddle.assign(self.baseline - (1.0 - self.decay) *
                          (self.baseline - self.rewards), self.baseline)
            self.loss = self.sample_log_probs * (self.rewards - self.baseline)
            clip = fluid.clip.GradientClipByNorm(clip_norm=5.0)
            if self.decay_steps is not None:
                lr = paddle.optimizer.lr.ExponentialDecay(
                    learning_rate=self.controller_lr,
                    gamma=self.decay_rate,
                    verbose=False)
            else:
                lr = self.controller_lr
            optimizer = paddle.optimizer.Adam(learning_rate=lr, grad_clip=clip)
            optimizer.minimize(self.loss)

    def _create_input(self, is_test=True, actual_rewards=None):
        feed_dict = dict()
        np_init_hidden = np.zeros(
            (self.controller_batch_size, self.hidden_size)).astype('float32')
        np_init_cell = np.zeros(
            (self.controller_batch_size, self.hidden_size)).astype('float32')

        feed_dict["hidden"] = np_init_hidden
        feed_dict["cell"] = np_init_cell

        if is_test == False:
            if isinstance(actual_rewards, np.float32):
                assert actual_rewards != None, "if you want to update controller, you must inputs a reward"
                actual_rewards = np.expand_dims(actual_rewards, axis=0)
            elif isinstance(actual_rewards, np.float) or isinstance(
                    actual_rewards, np.float64):
                actual_rewards = np.float32(actual_rewards)
                assert actual_rewards != None, "if you want to update controller, you must inputs a reward"
                actual_rewards = np.expand_dims(actual_rewards, axis=0)
            else:
                assert actual_rewards.all(
                ) != None, "if you want to update controller, you must inputs a reward"
                actual_rewards = actual_rewards.astype(np.float32)

            feed_dict['rewards'] = actual_rewards
            feed_dict['init_actions'] = np.array(self.init_tokens).astype(
                'int64')

        return feed_dict

    def next_tokens(self, num_archs=1, params_dict=None, is_inference=False):
        """ sample next tokens according current parameter and inputs"""
        self.num_archs = num_archs

        self.set_params(self.pred_program, params_dict, self.place)

        batch_tokens = []
        feed_dict = self._create_input()

        for _ in range(
                int(np.ceil(float(num_archs) / self.controller_batch_size))):
            if is_inference:
                self._build_program(is_inference=True)

            actions = self.exe.run(self.pred_program,
                                   feed=feed_dict,
                                   fetch_list=self.tokens)

            for idx in range(self.controller_batch_size):
                each_token = {}
                for i, action in enumerate(actions):
                    token = action[idx]
                    if idx in each_token:
                        each_token[idx].append(int(token))
                    else:
                        each_token[idx] = [int(token)]
                batch_tokens.append(each_token[idx])

        self.init_tokens = batch_tokens
        mod_token = (self.controller_batch_size -
                     (num_archs % self.controller_batch_size)
                     ) % self.controller_batch_size
        if mod_token != 0:
            return batch_tokens[:-mod_token]
        else:
            return batch_tokens

    def update(self, rewards, params_dict=None):
        """train controller according reward"""
        self.set_params(self.learn_program, params_dict, self.place)

        feed_dict = self._create_input(is_test=False, actual_rewards=rewards)

        loss = self.exe.run(self.learn_program,
                            feed=feed_dict,
                            fetch_list=[self.loss])
        _logger.info("Controller: current reward is {}, loss is {}".format(
            rewards, loss))
        params_dict = self.get_params(self.learn_program)
        return params_dict
