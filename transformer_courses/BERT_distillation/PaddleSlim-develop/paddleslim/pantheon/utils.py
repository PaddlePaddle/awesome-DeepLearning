#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections

public_authkey = u"aBcXyZ123"


class StartSignal():
    pass


class EndSignal():
    pass


class SyncSignal():
    pass


def convert_dtype(dtype):
    import paddle.fluid as fluid
    if isinstance(dtype, fluid.core.VarDesc.VarType):
        if dtype == fluid.core.VarDesc.VarType.BOOL:
            return 'bool'
        elif dtype == fluid.core.VarDesc.VarType.FP16:
            return 'float16'
        elif dtype == fluid.core.VarDesc.VarType.FP32:
            return 'float32'
        elif dtype == fluid.core.VarDesc.VarType.FP64:
            return 'float64'
        elif dtype == fluid.core.VarDesc.VarType.INT8:
            return 'int8'
        elif dtype == fluid.core.VarDesc.VarType.INT16:
            return 'int16'
        elif dtype == fluid.core.VarDesc.VarType.INT32:
            return 'int32'
        elif dtype == fluid.core.VarDesc.VarType.INT64:
            return 'int64'
        elif dtype == fluid.core.VarDesc.VarType.UINT8:
            return 'uint8'


def check_ip(address):
    import IPy
    try:
        IPy.IP(address)
        return True
    except Exception as e:
        return False
