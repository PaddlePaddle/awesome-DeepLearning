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

__all__ = ['AdaptiveNoiseSpec']


class AdaptiveNoiseSpec(object):
    def __init__(self):
        self.stdev_curr = 1.0

    def reset(self):
        self.stdev_curr = 1.0

    def update(self, action_dist):
        if action_dist > 1e-2:
            self.stdev_curr /= 1.03
        else:
            self.stdev_curr *= 1.03
