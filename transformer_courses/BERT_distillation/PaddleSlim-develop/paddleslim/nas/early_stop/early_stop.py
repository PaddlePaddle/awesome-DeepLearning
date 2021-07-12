#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

__all__ = ['EarlyStopBase']


class EarlyStopBase(object):
    """ Abstract early Stop algorithm.
    """

    def get_status(self, iter, result):
        """Get experiment status.
        """
        raise NotImplementedError(
            'get_status in Early Stop algorithm NOT implemented.')

    def client_end(self):
        """ Stop a client, this function may useful for the client that result is better and better.
        """
        raise NotImplementedError(
            'client_end in Early Stop algorithm NOT implemented.')
