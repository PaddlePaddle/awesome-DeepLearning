"""
config_utils
"""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import ast

import logger

logger = logger.Logger()

CONFIG_SECS = [
    'train',
    'valid',
    'test',
    'infer',
]

class AttrDict(dict):
    """
    AttrDict
    """
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    import yaml
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.Loader))
    create_attr_dict(yaml_config)
    return yaml_config


def create_attr_dict(yaml_config):
    """create_attr_dict"""
    for key, value in yaml_config.items():
        if isinstance(value, dict):
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value
    return


def print_configs(cfg, mode):
    """print_configs"""
    logger.info("---------------- {:>5} Arguments ----------------".format(
        mode))
    for sec, sec_items in cfg.items():
        logger.info("{}:".format(sec))
        for k, v in sec_items.items():
            logger.info("    {}:{}".format(k, v))
    logger.info("-------------------------------------------------")
