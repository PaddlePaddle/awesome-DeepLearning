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
import paddle
import pickle 
import numpy as np

usr_file = "./ml-1m/users.dat"
usr_info = {}
# 打开文件，读取所有行到data中
with open(usr_file, 'r') as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        usr_info[str(item[0])] = item

# 电影特征的路径
movie_data_path = "./ml-1m/movies.dat"
mov_info = {}
# 打开电影数据文件，根据电影ID索引到电影信息
with open(movie_data_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        mov_info[str(item[0])] = item

usr_id = 2
mov_id = 1

mov_feat_dir = 'mov_feat.pkl'
mov_feats = pickle.load(open(mov_feat_dir, 'rb'))

# 根据用户ID获得该用户的特征
usr_ID = 2
# 读取保存的用户特征
usr_feat_dir = 'usr_feat.pkl'
usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
# 根据用户ID索引到该用户的特征
usr_ID_feat = usr_feats[str(usr_ID)]

# 记录计算的相似度
cos_sims = []
# 记录下与用户特征计算相似的电影顺序
paddle.disable_static()
# 索引电影特征，计算和输入用户ID的特征的相似度
for idx, key in enumerate(mov_feats.keys()):
    mov_feat = mov_feats[key]
    usr_feat = paddle.to_tensor(usr_ID_feat)
    mov_feat = paddle.to_tensor(mov_feat)
    
    # 计算余弦相似度
    sim = paddle.nn.functional.common.cosine_similarity(usr_feat, mov_feat)
    # 打印特征和相似度的形状
    if idx==0:
        print("电影特征形状：{}, 用户特征形状：{}, 相似度结果形状：{}，相似度结果：{}".format(mov_feat.shape, usr_feat.shape, sim.numpy().shape, sim.numpy()))
    # 从形状为（1，1）的相似度sim中获得相似度值sim.numpy()[0]，并添加到相似度列表cos_sims中
    cos_sims.append(sim.numpy()[0])

# 对相似度排序，获得最大相似度在cos_sims中的位置
index = np.argsort(cos_sims)
# 打印相似度最大的前topk个位置
topk = 5
print("相似度最大的前{}个索引是{}\n对应的相似度是：{}\n".format(topk, index[-topk:], [cos_sims[k] for k in index[-topk:]]))
for i in index[-topk:]:
    print("对应的电影分别是：movie:{}".format(mov_info[list(mov_feats.keys())[i]]))





top_k, pick_num = 10, 6
# 对相似度排序，获得最大相似度在cos_sims中的位置
index = np.argsort(cos_sims)[-top_k:]

print("当前的用户是：")
# usr_id, usr_info 是前面定义、读取的用户ID、用户信息
print("usr_id:", usr_id, usr_info[str(usr_id)])   
print("推荐可能喜欢的电影是：")
res = []

# 加入随机选择因素，确保每次推荐的结果稍有差别
while len(res) < pick_num:
    val = np.random.choice(len(index), 1)[0]
    idx = index[val]
    mov_id = list(mov_feats.keys())[idx]
    if mov_id not in res:
        res.append(mov_id)

for id in res:
    print("mov_id:", id, mov_info[str(id)])