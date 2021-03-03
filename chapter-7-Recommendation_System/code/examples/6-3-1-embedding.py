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

# 声明用户的最大ID，在此基础上加1（算上数字0）
USR_ID_NUM = 6040 + 1
# 声明Embedding 层，将ID映射为32长度的向量
usr_emb = Embedding(num_embeddings=USR_ID_NUM,
                    embedding_dim=32,
                    sparse=False)
# 声明输入数据，将其转成tensor
arr_1 = np.array([1], dtype="int64").reshape((-1))
print(arr_1)
arr_pd1 = paddle.to_tensor(arr_1)
print(arr_pd1)
# 计算结果
emb_res = usr_emb(arr_pd1)
# 打印结果
print("数字 1 的embedding结果是： ", emb_res.numpy(), "\n形状是：", emb_res.shape)


# 声明用户的最大ID，在此基础上加1（算上数字0）
USR_ID_NUM = 10
# 声明Embedding 层，将ID映射为16长度的向量
usr_emb = Embedding(num_embeddings=USR_ID_NUM,
                    embedding_dim=16,
                    sparse=False)
# 定义输入数据，输入数据为不超过10的整数，将其转成tensor
arr = np.random.randint(0, 10, (3)).reshape((-1)).astype('int64')
print("输入数据是：", arr)
arr_pd = paddle.to_tensor(arr)
emb_res = usr_emb(arr_pd)
print("默认权重初始化embedding层的映射结果是：", emb_res.numpy())

# 观察Embedding层的权重
emb_weights = usr_emb.state_dict()
print(emb_weights.keys())

print("\n查看embedding层的权重形状：", emb_weights['weight'].shape)

# 声明Embedding 层，将ID映射为16长度的向量，自定义权重初始化方式
# 定义KaimingNorma初始化方式
init = paddle.nn.initializer.KaimingNormal()
param_attr = paddle.ParamAttr(initializer=init)

usr_emb2 = Embedding(num_embeddings=USR_ID_NUM,
                    embedding_dim=16,
                    weight_attr=param_attr)
emb_res = usr_emb2(arr_pd)
print("\KaimingNormal初始化权重embedding层的映射结果是：", emb_res.numpy())

