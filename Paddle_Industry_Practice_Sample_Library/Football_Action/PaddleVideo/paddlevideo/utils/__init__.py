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

from .registry import Registry
from .build_utils import build
from .config import *
from .logger import setup_logger, coloring, get_logger
from .record import AverageMeter, build_record, log_batch, log_epoch
from .dist_utils import get_dist_info, main_only
from .save_load import save, load, load_ckpt, mkdir
from .precise_bn import do_preciseBN
from .profiler import add_profiler_step
__all__ = ['Registry', 'build']
