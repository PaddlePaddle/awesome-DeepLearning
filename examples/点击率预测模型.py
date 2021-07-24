#!/usr/bin/env python
# coding: utf-8

# 作业1：设计完成两个推荐系统，根据电影推荐电影和根据用户推荐电影，并分析三个推荐系统的推荐结果差异
# 
# 作业2：构建一个【热门】、【新品】和【个性化推荐】三条推荐路径的混合系统（每次推荐10条，三种各占比例2、3、5条，每次的推荐结果不同）
# 
# 作业3：推荐系统的案例，实现本地的版本（非Aistudio上实现），进行训练和预测并截图提交
# 

# In[2]:


#!unzip -d /home/aistudio/work/ /home/aistudio/work/ml-1m.zip
#!unzip -d /home/aistudio/work/ /home/aistudio/work/save_feat.zip


# **作业1：设计完成两个推荐系统，根据电影推荐电影和根据用户推荐电影，并分析三个推荐系统的推荐结果差异 （答案）**

# In[3]:


#(1)根据用户喜欢（打5分）的电影推荐电影:
#首先，因为usr_feat和mov_feat的参数都是200维，所以将recommend_mov_for_usr（）函数的参数usr_id和
#usr_feat_dir换成mov_id和mov_feat_dir即可用此函数通过电影计算出相似的电影前top_k部。因为这是中间
#结果无需随机选取，所以此处pick_num的取值与top_k相同。通过运行发现推荐的电影与输入的电影类别相似且
#包括输入的样本电影本身。用户已看过的电影在下一步中过滤。


# 定义根据用户兴趣推荐电影
import pickle 
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

def recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
    assert pick_num <= top_k
    # 读取电影和用户的特征
    usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
    usr_feat = usr_feats[str(usr_id)]

    cos_sims = []

    with dygraph.guard():
        # 索引电影特征，计算和输入用户ID的特征的相似度
        for idx, key in enumerate(mov_feats.keys()):
            mov_feat = mov_feats[key]
            usr_feat = dygraph.to_variable(usr_feat)
            mov_feat = dygraph.to_variable(mov_feat)
            sim = fluid.layers.cos_sim(usr_feat, mov_feat)
            cos_sims.append(sim.numpy()[0][0])
    # 对相似度排序
    index = np.argsort(cos_sims)[-top_k:]

    mov_info = {}
    # 读取电影文件里的数据，根据电影ID索引到电影信息
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item
            
    #print("当前的用户是：")
    #print("usr_id:", usr_id)
    #print("推荐可能喜欢的电影是：")
    res = []
    
    # 加入随机选择因素，确保每次推荐的都不一样
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)

    #for id in res:
    #    print("mov_id:", id, mov_info[str(id)])
    return res

#测试通过用户查电影
movie_data_path = "./work/ml-1m/movies.dat"
top_k, pick_num = 10, 6
usr_id = 2
res = recommend_mov_for_usr(usr_id, top_k, pick_num, './work/usr_feat.pkl', './work/mov_feat.pkl', movie_data_path)
print('测试通过用户查电影:', '用户id：', usr_id, '    电影id：', res)

#测试通过电影查电影
movie_data_path = "./work/ml-1m/movies.dat"
top_k = 6
mov_id = 2
res = recommend_mov_for_usr(mov_id, pick_num, pick_num, './work/mov_feat.pkl', './work/mov_feat.pkl', movie_data_path)
print('测试通过电影查电影:', '电影id：', mov_id, '    电影id：', res)

#测试通过用户查用户
movie_data_path = "./work/ml-1m/movies.dat"
pick_num = 6
usr_id = 2
res = recommend_mov_for_usr(usr_id, pick_num, pick_num, './work/usr_feat.pkl', './work/usr_feat.pkl', movie_data_path)
print('测试通过用户查用户:', '用户id：', usr_id, '    用户id：', res)



# In[4]:


#根据用户看过的电影推荐相似的电影
import numpy as np
def recommend_mov_for_usr_m2m(u_id, pick_num, mov_feat_dir, rating_path):
    # 读取用户喜欢的电影（评分为5）
    with open(rating_path, 'r') as f:
        ratings_data = f.readlines()
    usr_rating_info = {}
    usr_favorite_info = {}
    for item in ratings_data:
        item = item.strip().split("::")
        # 处理每行数据，分别得到用户ID，电影ID，和评分
        usr_id,movie_id,score = item[0],item[1],item[2]
        if usr_id == str(u_id):
            usr_rating_info[movie_id] = score
            if score == '5':
                usr_favorite_info[movie_id] = score

    # 获得评分过的电影ID
    movie_ids_rated = list(usr_rating_info.keys())
    #print("ID为 {} 的用户，评分过的电影数量是: ".format(usr_id), len(movie_ids_rated))
    # 获得评分为5的电影ID
    movie_ids_5score = list(usr_favorite_info.keys())
    #print("ID为 {} 的用户，评分为5电影数量是: ".format(usr_id), len(movie_ids_5score))
    movie_ids_5score = np.array(movie_ids_5score)
    np.random.shuffle(movie_ids_5score)
    recommend_list = []
    for m_id in movie_ids_5score:
        res = recommend_mov_for_usr(m_id, pick_num, pick_num, mov_feat_dir, mov_feat_dir, movie_data_path)
        for m_id_rec in res:
            if m_id_rec not in movie_ids_rated: #推荐的电影没看过
                #print(m_id_rec)
                recommend_list.append(m_id_rec)
                if len(recommend_list) >= pick_num:
                    return recommend_list

usr_id = 2
res = recommend_mov_for_usr_m2m(usr_id, 10, './work/mov_feat.pkl', "./work/ml-1m/ratings.dat")
print("根据用户喜欢的电影推荐的电影：", res)
mov_info = {}
# 读取电影文件里的数据，根据电影ID索引到电影信息
with open("./work/ml-1m/movies.dat", 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        mov_info[str(item[0])] = item
for m in res:
    print("mov_id:", m, mov_info[str(m)])


# In[5]:


#（2）根据相似的用户喜欢（打5分）的电影推荐电影
#首先通过求余弦相似度查找电影品味相似的用户，这一步与前述一样通过更改输入参数复用recommend_mov_for_usr（）函数，
#求相似的用户。然后在得到的相似用户打5分的电影中选取推荐。
import numpy as np
def recommend_mov_for_usr_p2p(u_id, pick_num, usr_feat_dir, rating_path):
    # 读取用户喜欢的电影（评分为5）
    with open(rating_path, 'r') as f:
        ratings_data = f.readlines()
    usr_rating_info = {}
    for item in ratings_data:
        item = item.strip().split("::")
        # 处理每行数据，分别得到用户ID，电影ID，和评分
        usr_id,movie_id,score = item[0],item[1],item[2]
        if usr_id == str(u_id):
            usr_rating_info[movie_id] = score
    # 获得评分过的电影ID
    movie_ids_rated = list(usr_rating_info.keys())
    #print("ID为 {} 的用户，评分过的电影数量是: ".format(usr_id), len(movie_ids_rated))

    #得到电影品味相似的用户
    res = recommend_mov_for_usr(u_id, pick_num, pick_num, usr_feat_dir, usr_feat_dir, movie_data_path)
    usr_list = np.array(res)
    np.random.shuffle(usr_list)
    usr_favorite_info = {}
    for usr_id_simu in usr_list:
        for item in ratings_data:
            item = item.strip().split("::")
            # 处理每行数据，分别得到用户ID，电影ID，和评分
            usr_id,movie_id,score = item[0],item[1],item[2]
            # 得到相似用户评分为5的且被推荐用户未看过（打过分）的电影
            if usr_id == str(usr_id_simu) and score == '5' and movie_id not in movie_ids_rated:
                usr_favorite_info[movie_id] = score
        #的到5倍的候选电影列表后，随机选取前pick_num部电影推荐给用户
        if len(usr_favorite_info) >= pick_num * 5:
            break
    m_ids = list(usr_favorite_info.keys())
    m_ids = np.array(m_ids)
    np.random.shuffle(m_ids)
    if m_ids.shape[0] > pick_num:
        return m_ids[:pick_num]
    else:
        return m_ids

usr_id = 2
res = recommend_mov_for_usr_p2p(usr_id, 10, './work/usr_feat.pkl', "./work/ml-1m/ratings.dat")
print("根据用户喜欢的电影推荐的电影：", res)
mov_info = {}
# 读取电影文件里的数据，根据电影ID索引到电影信息
with open("./work/ml-1m/movies.dat", 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        mov_info[str(item[0])] = item
for m in res:
    print("mov_id:", m, mov_info[str(m)])


# 通过多次运行观察“根据电影推荐电影”和“根据用户推荐电影”两种方式与直接“根据用户与电影相似度”推荐方式相比较得出结论：本次作业完成的这两种间接推荐的方式选出的电影类别更加宽泛一些，与直接推荐的方式并无巨大差别。

# **作业2：构建一个【热门】、【新品】和【个性化推荐】三条推荐路径的混合系统（每次推荐10条，三种各占比例2、3、5条，每次的推荐结果不同）    答案**

# In[6]:


import numpy as np
def recommend_mov_for_usr_multi(usr_id, hot_num, new_num, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path, rating_path):
    # 选取热门电影（评分为5）2部
    #一部电影被打五分多少次就会被加入备选列表多少次，从而增加shuffle后被选中的概率
    with open(rating_path, 'r') as f:
        ratings_data = f.readlines()
    best_rating_info = {}
    for item in ratings_data:
        item = item.strip().split("::")
        # 处理每行数据，分别得到用户ID，电影ID，和评分
        usr_id,movie_id,score = item[0],item[1],item[2]
        if score == '5':
            best_rating_info[movie_id] = score
    # 获得评分为5的电影ID
    best_rating_info = list(best_rating_info.keys())
    best_rating_info = np.array(best_rating_info)
    np.random.shuffle(best_rating_info)
    res1 = best_rating_info[0:hot_num]

    res2 = []
    #选取新品电影（年份较新）3部
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            item[0] = int(item[0])
            item[1] = int(item[1][-5:-1])
            res2.append(item[0:2][:])
    res2 = np.array(res2)
    np.random.shuffle(res2)
    res2 = res2[np.argsort(-res2[:, 1])]
    res2 = res2[0:new_num, 0:1].T[0].astype(str)
    
    #选取个性化推荐的电影5部
    res3 = recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path)

    #拼接三种推荐结果并打乱次序
    res = np.concatenate([res1, res2, res3])
    np.random.shuffle(res)

    return res

usr_id = 2
res = recommend_mov_for_usr_multi(usr_id, 2, 3, 50, 5, './work/usr_feat.pkl', './work/mov_feat.pkl', "./work/ml-1m/movies.dat", "./work/ml-1m/ratings.dat")
print("根据【热门】、【新品】和【个性化推荐】推荐的电影：", res)
mov_info = {}
# 读取电影文件里的数据，根据电影ID索引到电影信息
with open("./work/ml-1m/movies.dat", 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    for item in data:
        item = item.strip().split("::")
        mov_info[str(item[0])] = item
for m in res:
    print("mov_id:", m, mov_info[str(m)])


# **作业3：推荐系统的案例，实现本地的版本（非Aistudio上实现），进行训练和预测并截图提交    答案**
# ![](https://ai-studio-static-online.cdn.bcebos.com/3ecc5ed21a1c480fad55ca9c9dc7635fa698c963356b4f76b5b40ca8f831160b)
