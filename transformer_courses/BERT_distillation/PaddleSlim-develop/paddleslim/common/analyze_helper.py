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
import types
import paddle
import paddle.fluid as fluid
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging
from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)


class Averager(object):
    def __init__(self):
        self.shadow = {}
        self.cnt = 0

    def register(self, name, val):
        self.shadow[name] = val
        self.cnt = 1

    def get(self, name):
        return self.shadow[name]

    def record(self):
        return self.shadow

    def update(self, name, val):
        assert name in self.shadow
        new_average = (self.cnt * self.shadow[name] + val) / (self.cnt + 1)
        self.cnt += 1
        self.shadow[name] = new_average


class EMA(Averager):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def update(self, name, val):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * val + self.decay * self.shadow[name]
        self.shadow[name] = new_average


class VarCollector(object):
    def __init__(self,
                 program,
                 var_names,
                 use_ema=False,
                 ema_decay=0.999,
                 scope=None):
        self.program = program
        self.var_names = var_names
        self.scope = paddle.static.global_scope() if scope is None else scope
        self.use_ema = use_ema
        self.set_up()
        if self.use_ema:
            self.stats = EMA(decay=ema_decay)
        else:
            self.stats = Averager()

    def set_up(self):
        self.real_names = []
        if hasattr(self.program, '_program'):
            program = self.program._program
        else:
            program = self.program

        for var in program.list_vars():
            if var.name in self.var_names:
                self.real_names.append(var.name)

    def update(self, vars_np):
        for name in self.real_names:
            val = vars_np[name]
            if val is not None:
                try:
                    self.stats.update(name, val)
                except:
                    self.stats.register(name, val)
            else:
                _logger.info("can't find var {}.".format(name))
        return self.stats.record()

    def run(self, reader, exe, step=None, loss_name=None):
        if not hasattr(self.program, '_program'):
            # Compile the native program to speed up
            program = paddle.static.CompiledProgram(
                self.program).with_data_parallel(loss_name=loss_name)

        for idx, data in enumerate(reader):
            vars_np = exe.run(program=program,
                              feed=data,
                              fetch_list=self.real_names)
            mapped_vars_np = dict(zip(self.real_names, vars_np))
            values = self.update(mapped_vars_np)

            if idx % 10 == 0:
                _logger.info("Collecting..., Step: {}".format(idx))
            if step is not None and idx + 1 >= step:
                break
        return values

    def abs_max_run(self, reader, exe, step=None, loss_name=None):
        fetch_list = []
        with paddle.static.program_guard(self.program):
            for act_name in self.real_names:
                act = self.program.global_block().var(act_name)
                act = paddle.max(paddle.abs(act), name=act_name + "_reduced")
                fetch_list.append(act_name + "_reduced.tmp_0")

        if not hasattr(self.program, '_program'):
            # Compile the native program to speed up
            program = paddle.static.CompiledProgram(
                self.program).with_data_parallel(loss_name=loss_name)
        for idx, data in enumerate(reader):
            vars_np = exe.run(program=program, feed=data, fetch_list=fetch_list)
            vars_np = [np.max(var) for var in vars_np]
            mapped_vars_np = dict(zip(self.real_names, vars_np))
            values = self.update(mapped_vars_np)

            if idx % 10 == 0:
                _logger.info("Collecting..., Step: {}".format(idx))

            if step is not None and idx + 1 >= step:
                break
        return values

    @staticmethod
    def pdf(var_dist, save_dir='dist_pdf'):
        """
        Draw histogram for distributtion of variables in that in var_dist.

        Args:
            var_dist(dict): numpy array of variables distribution.
            save_dir(str): dirname to save pdf. Default is 'dist_pdf'
        """
        numbers = len(var_dist)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            pdf_path = os.path.join(save_dir, 'result.pdf')
            with PdfPages(pdf_path) as pdf:
                for i, name in enumerate(var_dist.keys()):
                    if i % 10 == 0:
                        _logger.info("plt {}/{}".format(i, numbers))
                    arr = var_dist[name]
                    arr = arr.flatten()
                    weights = np.ones_like(arr) / len(arr)
                    plt.hist(arr, bins=1000, weights=weights)
                    plt.xlabel(name)
                    plt.ylabel("frequency")
                    plt.title("Hist of variable {}".format(name))
                    plt.show()
                    pdf.savefig()
                    plt.close()
        _logger.info("variables histogram have been saved as {}".format(
            pdf_path))
