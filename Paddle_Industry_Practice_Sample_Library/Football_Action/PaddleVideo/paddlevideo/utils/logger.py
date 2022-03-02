#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import logging
import os
import sys
import datetime

from paddle.distributed import ParallelEnv



Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('COLORING', True):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message


logger_initialized = []


def setup_logger(output=None, name="paddlevideo", level="INFO"):
    """
    Initialize the paddlevideo logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    def time_zone(sec, fmt):
        real_time = datetime.datetime.now()
        return real_time.timetuple()
    logging.Formatter.converter = time_zone

    logger = logging.getLogger(name)
    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level=="DEBUG":
        logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if level == "DEBUG":
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    local_rank = ParallelEnv().local_rank
    if local_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, ".log.txt")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)

        # PathManager.mkdirs(os.path.dirname(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # fh = logging.StreamHandler(_cached_log_stream(filename)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    logger_initialized.append(name)
    return logger


def get_logger(name, output=None):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    return setup_logger(name=name, output=name)
