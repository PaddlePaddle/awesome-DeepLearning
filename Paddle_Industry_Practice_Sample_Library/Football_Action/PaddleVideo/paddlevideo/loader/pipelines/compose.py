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

from collections.abc import Sequence
from ..registry import PIPELINES
import traceback
from ...utils import build
from ...utils import get_logger


@PIPELINES.register()
class Compose(object):
    """
    Composes several pipelines(include decode func, sample func, and transforms) together.

    Note: To deal with ```list``` type cfg temporaray, like:

        transform:
            - Crop: # A list
                attribute: 10
            - Resize: # A list
                attribute: 20

    every key of list will pass as the key name to build a module.
    XXX: will be improved in the future.

    Args:
        pipelines (list): List of transforms to compose.
    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.
    """
    def __init__(self, pipelines):
        #assert isinstance(pipelines, Sequence)
        self.pipelines = []
        for p in pipelines.values():
            if isinstance(p, dict):
                p = build(p, PIPELINES)
                self.pipelines.append(p)
            elif isinstance(p, list):
                for t in p:
                    #XXX: to deal with old format cfg, ugly code here!
                    temp_dict = dict(name=list(t.keys())[0])
                    for all_sub_t in t.values():
                        if all_sub_t is not None:
                            temp_dict.update(all_sub_t) 
      
                    t = build(temp_dict, PIPELINES)
                    self.pipelines.append(t)
            elif callable(p):
                self.pipelines.append(p)
            else:
                raise TypeError(f'pipelines must be callable or a dict,'
                                f'but got {type(p)}')
    def __call__(self, data):
        for p in self.pipelines:
            try:
                data = p(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger = get_logger("paddlevideo")
                logger.info("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(p, e, str(stack_info)))
                raise e
        return data
