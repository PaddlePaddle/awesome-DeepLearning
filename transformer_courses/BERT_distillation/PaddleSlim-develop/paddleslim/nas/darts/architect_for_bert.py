# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable


class Architect(object):
    def __init__(self, model, eta, arch_learning_rate, place, unrolled):
        self.network_momentum = 0.9
        self.network_weight_decay = 1e-3
        self.eta = eta
        self.model = model
        self.optimizer = fluid.optimizer.Adam(
            arch_learning_rate,
            0.5,
            0.999,
            regularization=fluid.regularizer.L2Decay(1e-3),
            parameter_list=self.model.arch_parameters())
        self.place = place
        self.unrolled = unrolled
        if self.unrolled:
            self.unrolled_model = self.model.new()
            self.unrolled_model_params = [
                p for p in self.unrolled_model.parameters()
                if p.name not in [
                    a.name for a in self.unrolled_model.arch_parameters()
                ] and p.trainable
            ]
            self.unrolled_optimizer = fluid.optimizer.MomentumOptimizer(
                self.eta,
                self.network_momentum,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    self.network_weight_decay),
                parameter_list=self.unrolled_model_params)

    def step(self, train_data, valid_data, epoch):
        if self.unrolled:
            params_grads = self._backward_step_unrolled(train_data, valid_data)
            self.optimizer.apply_gradients(params_grads)
        else:
            loss = self._backward_step(valid_data, epoch)
            self.optimizer.minimize(loss)
        self.optimizer.clear_gradients()

    def _backward_step(self, valid_data, epoch):
        loss = self.model.loss(valid_data, epoch)
        loss[0].backward()
        return loss[0]

    def _backward_step_unrolled(self, train_data, valid_data):
        self._compute_unrolled_model(train_data)
        unrolled_loss = self.unrolled_model.loss(valid_data)

        unrolled_loss.backward()
        vector = [
            to_variable(param._grad_ivar().numpy())
            for param in self.unrolled_model_params
        ]
        arch_params_grads = [
            (alpha, to_variable(ualpha._grad_ivar().numpy()))
            for alpha, ualpha in zip(self.model.arch_parameters(),
                                     self.unrolled_model.arch_parameters())
        ]
        self.unrolled_model.clear_gradients()

        implicit_grads = self._hessian_vector_product(vector, train_data)
        for (p, g), ig in zip(arch_params_grads, implicit_grads):
            new_g = g - (ig * self.unrolled_optimizer.current_step_lr())
            g.value().get_tensor().set(new_g.numpy(), self.place)
        return arch_params_grads

    def _compute_unrolled_model(self, data):
        for x, y in zip(self.unrolled_model.parameters(),
                        self.model.parameters()):
            x.value().get_tensor().set(y.numpy(), self.place)
        loss = self.unrolled_model._loss(data)
        loss.backward()
        self.unrolled_optimizer.minimize(loss)
        self.unrolled_model.clear_gradients()

    def _hessian_vector_product(self, vector, data, r=1e-2):
        R = r * fluid.layers.rsqrt(
            fluid.layers.sum([
                fluid.layers.reduce_sum(fluid.layers.square(v)) for v in vector
            ]))

        model_params = [
            p for p in self.model.parameters()
            if p.name not in [a.name for a in self.model.arch_parameters()] and
            p.trainable
        ]
        for param, grad in zip(model_params, vector):
            param_p = param + grad * R
            param.value().get_tensor().set(param_p.numpy(), self.place)
        loss = self.model.loss(data)
        loss.backward()
        grads_p = [
            to_variable(param._grad_ivar().numpy())
            for param in self.model.arch_parameters()
        ]

        for param, grad in zip(model_params, vector):
            param_n = param - grad * R * 2
            param.value().get_tensor().set(param_n.numpy(), self.place)
        self.model.clear_gradients()

        loss = self.model.loss(data)
        loss.backward()
        grads_n = [
            to_variable(param._grad_ivar().numpy())
            for param in self.model.arch_parameters()
        ]
        for param, grad in zip(model_params, vector):
            param_o = param + grad * R
            param.value().get_tensor().set(param_o.numpy(), self.place)
        self.model.clear_gradients()
        arch_grad = [(p - n) / (2 * R) for p, n in zip(grads_p, grads_n)]
        return arch_grad
