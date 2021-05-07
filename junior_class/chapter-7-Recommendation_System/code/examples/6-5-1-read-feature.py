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


# unzip -o data/save_feature_v1.zip -d /home/aistudio/

import pickle 
import numpy as np

mov_feat_dir = 'mov_feat.pkl'
usr_feat_dir = 'usr_feat.pkl'

usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
mov_feats = pickle.load(open(mov_feat_dir, 'rb'))

usr_id = 2
usr_feat = usr_feats[str(usr_id)]

mov_id = 1
# 通过电影ID索引到电影特征
mov_feat = mov_feats[str(mov_id)]

# 电影特征的路径
movie_data_path = "./ml-1m/movies.dat"
mov_info = {}
# 打开电影数据文件，根据电影ID索引到电影信息
with open(movie_data_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        mov_info[str(item[0])] = item

usr_file = "./ml-1m/users.dat"
usr_info = {}
# 打开文件，读取所有行到data中
with open(usr_file, 'r') as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        usr_info[str(item[0])] = item

print("当前的用户是：")
print("usr_id:", usr_id, usr_info[str(usr_id)])   
print("对应的特征是：", usr_feats[str(usr_id)])

print("\n当前电影是：")
print("mov_id:", mov_id, mov_info[str(mov_id)])
print("对应的特征是：")
print(mov_feat)

