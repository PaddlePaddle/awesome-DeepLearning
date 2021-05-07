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

# 自定义两个电影名称数据
mov_title_data = np.array(((1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 
                            (2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))).reshape(2, 1, 15).astype('int64')
# 对电影名称做映射，紧接着FC和pool层
MOV_TITLE_DICT_SIZE = 1000 + 1
mov_title_emb = Embedding(num_embeddings=MOV_TITLE_DICT_SIZE, embedding_dim=32)
mov_title_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(2, 1), padding=0)
# 使用 3 * 3卷积层代替全连接层
mov_title_conv2 = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1, padding=0)

mov_title_data = paddle.to_tensor(mov_title_data)
print("电影名称数据的输入形状: ", mov_title_data.shape)
# 1. 通过Embedding映射电影名称数据；
mov_title_feat = mov_title_emb(mov_title_data)
print("输入通过Embedding层的输出形状: ", mov_title_feat.shape)
# 2. 对Embedding后的向量使用卷积层进一步提取特征；
mov_title_feat = F.relu(mov_title_conv(mov_title_feat))
print("第一次卷积之后的特征输出形状: ", mov_title_feat.shape)
mov_title_feat = F.relu(mov_title_conv2(mov_title_feat))
print("第二次卷积之后的特征输出形状: ", mov_title_feat.shape)

batch_size = mov_title_data.shape[0]
# 3. 最后对特征进行降采样，keepdim=False会让输出的维度减少，而不是用[2,1,1,32]的形式占位；
mov_title_feat = paddle.sum(mov_title_feat, axis=2, keepdim=False)
print("reduce_sum降采样后的特征输出形状: ", mov_title_feat.shape)

mov_title_feat = F.relu(mov_title_feat)
mov_title_feat = paddle.reshape(mov_title_feat, [batch_size, -1])
print("电影名称特征的最终特征输出形状：", mov_title_feat.shape)

print("\n计算的电影名称的特征是", mov_title_feat.numpy(), "\n其形状是：", mov_title_feat.shape)
print("\n电影名称为 {} 计算得到的特征是：{}".format(mov_title_data.numpy()[0,:, 0], mov_title_feat.numpy()[0]))
print("\n电影名称为 {} 计算得到的特征是：{}".format(mov_title_data.numpy()[1,:, 0], mov_title_feat.numpy()[1]))