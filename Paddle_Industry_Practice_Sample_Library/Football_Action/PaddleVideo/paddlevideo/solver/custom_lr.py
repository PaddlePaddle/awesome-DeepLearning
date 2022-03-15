# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import math
from paddle.optimizer.lr import *
"""
PaddleVideo Learning Rate Schedule:
You can use paddle.optimizer.lr
or define your custom_lr in this file.
"""


class CustomWarmupCosineDecay(LRScheduler):
    r"""
    We combine warmup and stepwise-cosine which is used in slowfast model.

    Args:
        warmup_start_lr (float): start learning rate used in warmup stage.
        warmup_epochs (int): the number epochs of warmup.
        cosine_base_lr (float|int, optional): base learning rate in cosine schedule.
        max_epoch (int): total training epochs.
        num_iters(int): number iterations of each epoch.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    Returns:
        ``CosineAnnealingDecay`` instance to schedule learning rate.
    """
    def __init__(self,
                 warmup_start_lr,
                 warmup_epochs,
                 cosine_base_lr,
                 max_epoch,
                 num_iters,
                 last_epoch=-1,
                 verbose=False):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_base_lr = cosine_base_lr
        self.max_epoch = max_epoch
        self.num_iters = num_iters
        #call step() in base class, last_lr/last_epoch/base_lr will be update
        super(CustomWarmupCosineDecay, self).__init__(last_epoch=last_epoch,
                                                      verbose=verbose)

    def step(self, epoch=None):
        """
        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .
        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        """
        if epoch is None:
            if self.last_epoch == -1:
                self.last_epoch += 1
            else:
                self.last_epoch += 1 / self.num_iters  # update step with iters
        else:
            self.last_epoch = epoch
        self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def _lr_func_cosine(self, cur_epoch, cosine_base_lr, max_epoch):
        return cosine_base_lr * (math.cos(math.pi * cur_epoch / max_epoch) +
                                 1.0) * 0.5

    def get_lr(self):
        """Define lr policy"""
        lr = self._lr_func_cosine(self.last_epoch, self.cosine_base_lr,
                                  self.max_epoch)
        lr_end = self._lr_func_cosine(self.warmup_epochs, self.cosine_base_lr,
                                      self.max_epoch)

        # Perform warm up.
        if self.last_epoch < self.warmup_epochs:
            lr_start = self.warmup_start_lr
            alpha = (lr_end - lr_start) / self.warmup_epochs
            lr = self.last_epoch * alpha + lr_start
        return lr


class CustomWarmupPiecewiseDecay(LRScheduler):
    r"""
    This op combine warmup and stepwise-cosine which is used in slowfast model.

    Args:
        warmup_start_lr (float): start learning rate used in warmup stage.
        warmup_epochs (int): the number epochs of warmup.
        step_base_lr (float|int, optional): base learning rate in step schedule.
        max_epoch (int): total training epochs.
        num_iters(int): number iterations of each epoch.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    Returns:
        ``CustomWarmupPiecewiseDecay`` instance to schedule learning rate.
    """
    def __init__(self,
                 warmup_start_lr,
                 warmup_epochs,
                 step_base_lr,
                 lrs,
                 gamma,
                 steps,
                 max_epoch,
                 num_iters,
                 last_epoch=0,
                 verbose=False):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.step_base_lr = step_base_lr
        self.lrs = lrs
        self.gamma = gamma
        self.steps = steps
        self.max_epoch = max_epoch
        self.num_iters = num_iters
        self.last_epoch = last_epoch
        self.last_lr = self.warmup_start_lr  # used in first iter
        self.verbose = verbose
        self._var_name = None

    def step(self, epoch=None, rebuild=False):
        """
        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .
        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        """
        if epoch is None:
            if not rebuild:
                self.last_epoch += 1 / self.num_iters  # update step with iters
        else:
            self.last_epoch = epoch
        self.last_lr = self.get_lr()

        if self.verbose:
            print(
                'step Epoch {}: {} set learning rate to {}.self.num_iters={}, 1/self.num_iters={}'
                .format(self.last_epoch, self.__class__.__name__, self.last_lr,
                        self.num_iters, 1 / self.num_iters))

    def _lr_func_steps_with_relative_lrs(self, cur_epoch, lrs, base_lr, steps,
                                         max_epoch):
        # get step index
        steps = steps + [max_epoch]
        for ind, step in enumerate(steps):
            if cur_epoch < step:
                break
        if self.verbose:
            print(
                '_lr_func_steps_with_relative_lrs, cur_epoch {}: {}, steps {}, ind {}, step{}, max_epoch{}'
                .format(cur_epoch, self.__class__.__name__, steps, ind, step,
                        max_epoch))

        return lrs[ind - 1] * base_lr

    def get_lr(self):
        """Define lr policy"""
        lr = self._lr_func_steps_with_relative_lrs(
            self.last_epoch,
            self.lrs,
            self.step_base_lr,
            self.steps,
            self.max_epoch,
        )
        lr_end = self._lr_func_steps_with_relative_lrs(
            self.warmup_epochs,
            self.lrs,
            self.step_base_lr,
            self.steps,
            self.max_epoch,
        )

        # Perform warm up.
        if self.last_epoch < self.warmup_epochs:
            lr_start = self.warmup_start_lr
            alpha = (lr_end - lr_start) / self.warmup_epochs
            lr = self.last_epoch * alpha + lr_start
        if self.verbose:
            print(
                'get_lr, Epoch {}: {}, lr {}, lr_end {}, self.lrs{}, self.step_base_lr{}, self.steps{}, self.max_epoch{}'
                .format(self.last_epoch, self.__class__.__name__, lr, lr_end,
                        self.lrs, self.step_base_lr, self.steps,
                        self.max_epoch))

        return lr


class CustomPiecewiseDecay(PiecewiseDecay):
    def __init__(self, **kargs):
        kargs.pop('num_iters')
        super().__init__(**kargs)


class CustomWarmupCosineStepDecay(LRScheduler):
    def __init__(self,
                 warmup_iters,
                 warmup_ratio=0.1,
                 min_lr=0,
                 base_lr=3e-5,
                 max_epoch=30,
                 last_epoch=-1,
                 num_iters=None,
                 verbose=False):

        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        self.warmup_epochs = warmup_iters
        self.warmup_iters = warmup_iters * num_iters
        self.cnt_iters = 0
        self.cnt_epoch = 0
        self.num_iters = num_iters
        self.tot_iters = max_epoch * num_iters
        self.max_epoch = max_epoch
        self.cosine_base_lr = base_lr  # initial lr for all param groups
        self.regular_lr = self.get_regular_lr()
        super().__init__(last_epoch=last_epoch, verbose=verbose)

    def annealing_cos(self, start, end, factor, weight=1):
        cos_out = math.cos(math.pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_regular_lr(self):
        progress = self.cnt_epoch
        max_progress = self.max_epoch
        target_lr = self.min_lr
        return self.annealing_cos(self.cosine_base_lr, target_lr, progress /
                                  max_progress)  # self.cosine_base_lr

    def get_warmup_lr(self, cur_iters):
        k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
        warmup_lr = self.regular_lr * (1 - k)  # 3e-5 * (1-k)
        return warmup_lr

    def step(self, epoch=None):
        self.regular_lr = self.get_regular_lr()
        self.last_lr = self.get_lr()
        self.cnt_epoch = (self.cnt_iters +
                          1) // self.num_iters  # update step with iters
        self.cnt_iters += 1

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def get_lr(self):
        """Define lr policy"""
        cur_iter = self.cnt_iters
        if cur_iter >= self.warmup_iters:
            return self.regular_lr
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            return warmup_lr
