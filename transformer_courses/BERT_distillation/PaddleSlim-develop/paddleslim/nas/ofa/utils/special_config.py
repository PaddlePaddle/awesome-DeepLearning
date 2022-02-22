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

import numpy as np

__all__ = ['dynabert_config']


def dynabert_config(model, width_mult, depth_mult=1.0):
    new_config = dict()
    block_num = np.floor((len(model.layers.items()) - 3) / 6)
    block_name = block_num * 6 + 2

    def fix_exp(idx):
        if (idx - 3) % 6 == 0 or (idx - 5) % 6 == 0:
            return True
        return False

    start_idx = 0
    for idx, (block_k, block_v) in enumerate(model.layers.items()):
        if 'linear' in block_k:
            start_idx = int(block_k.split('_')[1])
            break

    for idx, (block_k, block_v) in enumerate(model.layers.items()):
        if isinstance(block_v, dict) and len(block_v.keys()) != 0:
            name, name_idx = block_k.split('_'), int(block_k.split('_')[1])
            if fix_exp(name_idx -
                       start_idx) or 'emb' in block_k or idx >= block_name:
                block_v['expand_ratio'] = 1.0
            else:
                block_v['expand_ratio'] = width_mult

        if block_k == 'depth':
            block_v = depth_mult

        if block_k != 'depth':
            new_block_k = model._key2name[block_k]
        else:
            new_block_k = 'depth'
        new_config[new_block_k] = block_v
    return new_config
