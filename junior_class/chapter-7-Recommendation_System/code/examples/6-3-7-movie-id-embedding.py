
# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
from paddle.nn import Linear, Embedding, Conv2D
import numpy as np
import paddle.nn.functional as F

# 自定义一个电影ID数据
mov_id_data = np.array((1, 2)).reshape(-1).astype('int64')
# 对电影ID信息做映射，并紧接着一个FC层
MOV_DICT_SIZE = 3952 + 1
mov_emb = Embedding(num_embeddings=MOV_DICT_SIZE, embedding_dim=32)
mov_fc = Linear(32, 32)


print("输入的电影ID是:", mov_id_data)
mov_id_data = paddle.to_tensor(mov_id_data)
mov_id_feat = mov_fc(mov_emb(mov_id_data))
mov_id_feat = F.relu(mov_id_feat)
print("计算的电影ID的特征是", mov_id_feat.numpy(), "\n其形状是：", mov_id_feat.shape)
print("\n电影ID为 {} 计算得到的特征是：{}".format(mov_id_data.numpy()[0], mov_id_feat.numpy()[0]))
print("电影ID为 {} 计算得到的特征是：{}".format(mov_id_data.numpy()[1], mov_id_feat.numpy()[1]))