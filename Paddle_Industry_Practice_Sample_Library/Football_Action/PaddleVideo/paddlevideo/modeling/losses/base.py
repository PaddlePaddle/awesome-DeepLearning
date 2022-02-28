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

from abc import  abstractmethod
import paddle
import paddle.nn as nn

#XXX use _forward?? or forward??
class BaseWeightedLoss(nn.Layer):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.
        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.
        Returns:
            paddle.Tensor: The calculated loss.
        """
        return self._forward(*args, **kwargs) * self.loss_weight
