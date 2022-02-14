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


def build(cfg, registry, key='name'):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key.
        registry (XXX): The registry to search the type from.
        key (str): the key.
    Returns:
        obj: The constructed object.
    """

    assert isinstance(cfg, dict) and key in cfg

    cfg_copy = cfg.copy()
    obj_type = cfg_copy.pop(key)

    obj_cls = registry.get(obj_type)
    if obj_cls is None:
        raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    return obj_cls(**cfg_copy)
