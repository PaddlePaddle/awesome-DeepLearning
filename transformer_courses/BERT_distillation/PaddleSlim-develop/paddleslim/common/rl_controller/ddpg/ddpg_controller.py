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

import numpy as np
import parl
from parl import layers
import paddle
from paddle import fluid
from ..utils import RLCONTROLLER, action_mapping
from ...controller import RLBaseController
from .ddpg_model import DefaultDDPGModel as default_ddpg_model
from .noise import AdaptiveNoiseSpec as default_noise
from parl.utils import ReplayMemory

__all__ = ['DDPG']


class DDPGAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(DDPGAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = paddle.static.Program()
        self.learn_program = paddle.static.Program()

        with paddle.static.program_guard(self.pred_program):
            obs = fluid.data(
                name='obs', shape=[None, self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with paddle.static.program_guard(self.learn_program):
            obs = fluid.data(
                name='obs', shape=[None, self.obs_dim], dtype='float32')
            act = fluid.data(
                name='act', shape=[None, self.act_dim], dtype='float32')
            reward = fluid.data(name='reward', shape=[None], dtype='float32')
            next_obs = fluid.data(
                name='next_obs', shape=[None, self.obs_dim], dtype='float32')
            terminal = fluid.data(
                name='terminal', shape=[None, 1], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        act = self.fluid_executor.run(self.pred_program,
                                      feed={'obs': obs},
                                      fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(self.learn_program,
                                              feed=feed,
                                              fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost


@RLCONTROLLER.register
class DDPG(RLBaseController):
    def __init__(self, range_tables, use_gpu=False, **kwargs):
        self.use_gpu = use_gpu
        self.range_tables = range_tables - np.asarray(1)
        self.act_dim = len(self.range_tables)
        self.obs_dim = kwargs.get('obs_dim')
        self.model = kwargs.get(
            'model') if 'model' in kwargs else default_ddpg_model
        self.actor_lr = kwargs.get('actor_lr') if 'actor_lr' in kwargs else 1e-4
        self.critic_lr = kwargs.get(
            'critic_lr') if 'critic_lr' in kwargs else 1e-3
        self.gamma = kwargs.get('gamma') if 'gamma' in kwargs else 0.99
        self.tau = kwargs.get('tau') if 'tau' in kwargs else 0.001
        self.memory_size = kwargs.get(
            'memory_size') if 'memory_size' in kwargs else 10
        self.reward_scale = kwargs.get(
            'reward_scale') if 'reward_scale' in kwargs else 0.1
        self.batch_size = kwargs.get(
            'controller_batch_size') if 'controller_batch_size' in kwargs else 1
        self.actions_noise = kwargs.get(
            'actions_noise') if 'actions_noise' in kwargs else default_noise
        self.action_dist = 0.0
        self.place = paddle.CUDAPlace(0) if self.use_gpu else paddle.CPUPlace()

        model = self.model(self.act_dim)

        if self.actions_noise:
            self.actions_noise = self.actions_noise()

        algorithm = parl.algorithms.DDPG(
            model,
            gamma=self.gamma,
            tau=self.tau,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr)
        self.agent = DDPGAgent(algorithm, self.obs_dim, self.act_dim)
        self.rpm = ReplayMemory(self.memory_size, self.obs_dim, self.act_dim)

        self.pred_program = self.agent.pred_program
        self.learn_program = self.agent.learn_program
        self.param_dict = self.get_params(self.learn_program)

    def next_tokens(self, obs, params_dict, is_inference=False):
        batch_obs = np.expand_dims(obs, axis=0)
        self.set_params(self.pred_program, params_dict, self.place)
        actions = self.agent.predict(batch_obs.astype('float32'))
        ### add noise to action
        if self.actions_noise and is_inference == False:
            actions_noise = np.clip(
                np.random.normal(
                    actions, scale=self.actions_noise.stdev_curr),
                -1.0,
                1.0)
            self.action_dist = np.mean(np.abs(actions_noise - actions))
        else:
            actions_noise = actions
        actions_noise = action_mapping(actions_noise, self.range_tables)
        return actions_noise

    def _update_noise(self, actions_dist):
        self.actions_noise.update(actions_dist)

    def update(self, rewards, params_dict, obs, actions, obs_next, terminal):
        self.set_params(self.learn_program, params_dict, self.place)
        self.rpm.append(obs, actions, self.reward_scale * rewards, obs_next,
                        terminal)
        if self.actions_noise:
            self._update_noise(self.action_dist)
        if self.rpm.size() > self.memory_size:
            obs, actions, rewards, obs_next, terminal = rpm.sample_batch(
                self.batch_size)
        self.agent.learn(obs, actions, rewards, obs_next, terminal)
        params_dict = self.get_params(self.learn_program)
        return params_dict
