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

from .combine_search_space import CombineSearchSpace

__all__ = ["SearchSpaceFactory"]


class SearchSpaceFactory(object):
    def __init__(self):
        pass

    def get_search_space(self, config_lists):
        """
        get model spaces based on list(key, config). 

        """
        assert isinstance(config_lists, list), "configs must be a list"

        return CombineSearchSpace(config_lists)
