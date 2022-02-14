"""
reader_util
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import random
import numpy as np


class ReaderNotFoundError(Exception):
    """
    "Error: reader not found"
    """

    def __init__(self, reader_name, avail_readers):
        super(ReaderNotFoundError, self).__init__()
        self.reader_name = reader_name
        self.avail_readers = avail_readers

    def __str__(self):
        msg = "Reader {} Not Found.\nAvailiable readers:\n".format(
            self.reader_name)
        for reader in self.avail_readers:
            msg += "  {}\n".format(reader)
        return msg


class DataReader(object):
    """
    data reader for video input
    """

    def __init__(self, model_name, mode, cfg):
        self.name = model_name
        self.mode = mode
        self.cfg = cfg

    def create_reader(self):
        """
        Not implemented
        """
        pass

    def get_config_from_sec(self, sec, item, default=None):
        """
        get_config_from_sec
        """
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)


class ReaderZoo(object):
    """
    ReaderZoo
    """
    def __init__(self):
        """
        __init__
        """
        self.reader_zoo = {}

    def regist(self, name, reader):
        """
        regist
        """
        assert reader.__base__ == DataReader, "Unknow model type {}".format(
            type(reader))
        self.reader_zoo[name] = reader

    def get(self, name, mode, cfg, material=None):
        """
        get
        """
        for k, v in self.reader_zoo.items():
            if k == name:
                return v(name, mode, cfg, material)
        raise ReaderNotFoundError(name, self.reader_zoo.keys())


# singleton reader_zoo
reader_zoo = ReaderZoo()


def regist_reader(name, reader):
    """
    regist_reader
    """
    reader_zoo.regist(name, reader)


def get_reader(name, mode, cfg, material=None):
    """
    get_reader
    """
    reader_model = reader_zoo.get(name, mode, cfg, material)
    return reader_model.create_reader()
