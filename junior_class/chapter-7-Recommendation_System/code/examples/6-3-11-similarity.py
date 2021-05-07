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

# 自定义一个用户ID数据
usr_id_data = np.random.randint(0, 6040, (2)).reshape((-1)).astype('int64')
print("输入的用户ID是:", usr_id_data)

USR_ID_NUM = 6040 + 1
# 定义用户ID的embedding层和fc层
usr_emb = Embedding(num_embeddings=USR_ID_NUM,
                embedding_dim=32,
                sparse=False)
usr_fc = Linear(in_features=32, out_features=32)

usr_id_var = paddle.to_tensor(usr_id_data)
usr_id_feat = usr_fc(usr_emb(usr_id_var))

usr_id_feat = F.relu(usr_id_feat)

# 自定义一个用户职业数据
usr_job_data = np.array((0, 20)).reshape(-1).astype('int64')
print("输入的用户职业是:", usr_job_data)

# 对用户职业信息做映射，并紧接着一个Linear层
# 用户职业的最大ID是20，所以Embedding层size的第一个参数设置为20 + 1 = 21
USR_JOB_DICT_SIZE = 20 + 1
usr_job_emb = Embedding(num_embeddings=USR_JOB_DICT_SIZE,embedding_dim=16)
usr_job_fc = Linear(in_features=16, out_features=16)

usr_job = paddle.to_tensor(usr_job_data)
usr_job_feat = usr_job_emb(usr_job)
usr_job_feat = usr_job_fc(usr_job_feat)
usr_job_feat = F.relu(usr_job_feat)

# 自定义一个用户年龄数据
usr_age_data = np.array((1, 18)).reshape(-1).astype('int64')
print("输入的用户年龄是:", usr_age_data)

# 对用户年龄信息做映射，并紧接着一个Linear层
# 年龄的最大ID是56，所以Embedding层size的第一个参数设置为56 + 1 = 57
USR_AGE_DICT_SIZE = 56 + 1

usr_age_emb = Embedding(num_embeddings=USR_AGE_DICT_SIZE,
                            embedding_dim=16)
usr_age_fc = Linear(in_features=16, out_features=16)

usr_age = paddle.to_tensor(usr_age_data)
usr_age_feat = usr_age_emb(usr_age)
usr_age_feat = usr_age_fc(usr_age_feat)
usr_age_feat = F.relu(usr_age_feat)

# 自定义一个用户性别数据
usr_gender_data = np.array((0, 1)).reshape(-1).astype('int64')
print("输入的用户性别是:", usr_gender_data)

# 用户的性别用0， 1 表示
# 性别最大ID是1，所以Embedding层size的第一个参数设置为1 + 1 = 2
USR_ID_NUM = 2
# 对用户性别信息做映射，并紧接着一个FC层
USR_GENDER_DICT_SIZE = 2
usr_gender_emb = Embedding(num_embeddings=USR_GENDER_DICT_SIZE,
                            embedding_dim=16)

usr_gender_fc = Linear(in_features=16, out_features=16)

usr_gender_var = paddle.to_tensor(usr_gender_data)
usr_gender_feat = usr_gender_fc(usr_gender_emb(usr_gender_var))
usr_gender_feat = F.relu(usr_gender_feat)

usr_combined = Linear(in_features=80, out_features=200)

# 收集所有的用户特征
_features = [usr_id_feat, usr_job_feat, usr_age_feat, usr_gender_feat]

print("打印每个特征的维度：", [f.shape for f in _features])

_features = [k.numpy() for k in _features]
_features = [paddle.to_tensor(k) for k in _features]

# 对特征沿着最后一个维度级联
usr_feat = paddle.concat(_features, axis=1)
usr_feat = F.tanh(usr_combined(usr_feat))


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

mov_combined = Linear(in_features=96, out_features=200)
# 收集所有的电影特征
_features = [mov_id_feat, mov_cat_feat, mov_title_feat]
_features = [k.numpy() for k in _features]
_features = [paddle.to_tensor(k) for k in _features]

# 对特征沿着最后一个维度级联
mov_feat = paddle.concat(_features, axis=1)
mov_feat = mov_combined(mov_feat)
mov_feat = F.tanh(mov_feat)

def similarty(usr_feature, mov_feature):
    res = F.common.cosine_similarity(usr_feature, mov_feature)
    res = paddle.scale(res, scale=5)
    return usr_feat, mov_feat, res

# 使用上文计算得到的用户特征和电影特征计算相似度
_sim = similarty(usr_feat, mov_feat)
print("相似度是：", np.squeeze(_sim[-1].numpy()))