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

FC_ID = Linear(in_features=32, out_features=200)
FC_JOB = Linear(in_features=16, out_features=200)
FC_AGE = Linear(in_features=16, out_features=200)
FC_GENDER = Linear(in_features=16, out_features=200)

# 自定义一个用户ID数据
usr_id_data = np.random.randint(0, 6040, (2)).reshape((-1)).astype('int64')
USR_ID_NUM = 6040 + 1
# 定义用户ID的embedding层和fc层
usr_emb = Embedding(num_embeddings=USR_ID_NUM,
                embedding_dim=32,
                sparse=False)
usr_fc = Linear(in_features=32, out_features=32)

usr_id_var = paddle.to_tensor(usr_id_data)
usr_id_feat = usr_fc(usr_emb(usr_id_var))

usr_id_feat = F.relu(usr_id_feat)

# 自定义一个用户年龄数据
usr_age_data = np.array((1, 18)).reshape(-1).astype('int64')
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


# 自定义一个用户职业数据
usr_job_data = np.array((0, 20)).reshape(-1).astype('int64')
# 用户职业的最大ID是20，所以Embedding层size的第一个参数设置为20 + 1 = 21
USR_JOB_DICT_SIZE = 20 + 1
usr_job_emb = Embedding(num_embeddings=USR_JOB_DICT_SIZE,embedding_dim=16)
usr_job_fc = Linear(in_features=16, out_features=16)

usr_job = paddle.to_tensor(usr_job_data)
usr_job_feat = usr_job_emb(usr_job)
usr_job_feat = usr_job_fc(usr_job_feat)
usr_job_feat = F.relu(usr_job_feat)

# 收集所有的用户特征
_features = [usr_id_feat, usr_job_feat, usr_age_feat, usr_gender_feat]
_features = [k.numpy() for k in _features]
_features = [paddle.to_tensor(k) for k in _features]

id_feat = F.tanh(FC_ID(_features[0]))
job_feat = F.tanh(FC_JOB(_features[1]))
age_feat = F.tanh(FC_AGE(_features[2]))
genger_feat = F.tanh(FC_GENDER(_features[-1]))

# 对特征求和
usr_feat = id_feat + job_feat + age_feat + genger_feat
print("用户融合后特征的维度是：", usr_feat.shape)
