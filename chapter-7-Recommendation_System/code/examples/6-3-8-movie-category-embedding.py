
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

# 自定义一个电影类别数据
mov_cat_data = np.array(((1, 2, 3, 0, 0, 0), (2, 3, 4, 0, 0, 0))).reshape(2, -1).astype('int64')
# 对电影ID信息做映射，并紧接着一个Linear层
MOV_DICT_SIZE = 6 + 1
mov_emb = Embedding(num_embeddings=MOV_DICT_SIZE, embedding_dim=32)
mov_fc = Linear(in_features=32, out_features=32)

print("输入的电影类别是:", mov_cat_data[:, :])
mov_cat_data = paddle.to_tensor(mov_cat_data)
# 1. 通过Embedding映射电影类别数据；
mov_cat_feat = mov_emb(mov_cat_data)
# 2. 对Embedding后的向量沿着类别数量维度进行求和，得到一个类别映射向量；
mov_cat_feat = paddle.sum(mov_cat_feat, axis=1, keepdim=False)

# 3. 通过一个全连接层计算类别特征向量。
mov_cat_feat = mov_fc(mov_cat_feat)
mov_cat_feat = F.relu(mov_cat_feat)
print("计算的电影类别的特征是", mov_cat_feat.numpy(), "\n其形状是：", mov_cat_feat.shape)
print("\n电影类别为 {} 计算得到的特征是：{}".format(mov_cat_data.numpy()[0, :], mov_cat_feat.numpy()[0]))
print("\n电影类别为 {} 计算得到的特征是：{}".format(mov_cat_data.numpy()[1, :], mov_cat_feat.numpy()[1]))