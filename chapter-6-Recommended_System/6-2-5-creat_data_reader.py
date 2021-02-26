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

import numpy as np
def get_usr_info(path):
    # 性别转换函数，M-0， F-1
    def gender2num(gender):
        return 1 if gender == 'F' else 0
    
    # 打开文件，读取所有行到data中
    with open(path, 'r') as f:
        data = f.readlines()
    # 建立用户信息的字典
    use_info = {}
    
    max_usr_id = 0
    #按行索引数据
    for item in data:
        # 去除每一行中和数据无关的部分
        item = item.strip().split("::")
        usr_id = item[0]
        # 将字符数据转成数字并保存在字典中
        use_info[usr_id] = {'usr_id': int(usr_id),
                            'gender': gender2num(item[1]),
                            'age': int(item[2]),
                            'job': int(item[3])}
        max_usr_id = max(max_usr_id, int(usr_id))
    
    return use_info, max_usr_id

usr_file = "./work/ml-1m/users.dat"
usr_info, max_usr_id = get_usr_info(usr_file)


def get_movie_info(path):
    # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中 
    with open(path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
    # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
    movie_info, movie_titles, movie_cat = {}, {}, {}
    # 对电影名字、类别中不同的单词计数
    t_count, c_count = 1, 1
    # 初始化电影名字和种类的列表
    titles = []
    cats = []
    count_tit = {}
    # 按行读取数据并处理
    for item in data:
        item = item.strip().split("::")
        v_id = item[0]
        v_title = item[1][:-7]
        cats = item[2].split('|')
        v_year = item[1][-5:-1]

        titles = v_title.split()
        # 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
        for t in titles:
            if t not in movie_titles:
                movie_titles[t] = t_count
                t_count += 1
        # 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
        for cat in cats:
            if cat not in movie_cat:
                movie_cat[cat] = c_count
                c_count += 1
        # 补0使电影名称对应的列表长度为15
        v_tit = [movie_titles[k] for k in titles]
        while len(v_tit)<15:
            v_tit.append(0)
        # 补0使电影种类对应的列表长度为6
        v_cat = [movie_cat[k] for k in cats]
        while len(v_cat)<6:
            v_cat.append(0)
        # 保存电影数据到movie_info中
        movie_info[v_id] = {'mov_id': int(v_id),
                            'title': v_tit,
                            'category': v_cat,
                            'years': int(v_year)}
    return movie_info, movie_cat, movie_titles


movie_info_path = "./work/ml-1m/movies.dat"
movie_info, movie_cat, movie_titles = get_movie_info(movie_info_path)


def get_rating_info(path):
    # 打开文件，读取所有行到data中
    with open(path, 'r') as f:
        data = f.readlines()
    # 创建一个字典
    rating_info = {}
    for item in data:
        item = item.strip().split("::")
        # 处理每行数据，分别得到用户ID，电影ID，和评分
        usr_id,movie_id,score = item[0],item[1],item[2]
        if usr_id not in rating_info.keys():
            rating_info[usr_id] = {movie_id:float(score)}
        else:
            rating_info[usr_id][movie_id] = float(score)
    return rating_info

# 获得评分数据
rating_path = "./work/ml-1m/ratings.dat"
rating_info = get_rating_info(rating_path)


def get_dataset(usr_info, rating_info, movie_info):
    trainset = []
    # 按照评分数据的key值索引数据
    for usr_id in rating_info.keys():
        usr_ratings = rating_info[usr_id]
        for movie_id in usr_ratings:
            trainset.append({'usr_info': usr_info[usr_id],
                             'mov_info': movie_info[movie_id],
                             'scores': usr_ratings[movie_id]})
    return trainset

dataset = get_dataset(usr_info, rating_info, movie_info)
print("数据集总数据数：", len(dataset))


import random
def load_data(dataset=None, mode='train'):
    
    """定义一些超参数等等"""
    
    # 定义数据迭代加载器
    def data_generator():
        
        """ 定义数据的处理过程"""
        
        data  = None
        yield data
        
    # 返回数据迭代加载器
    return data_generator

import random
use_poster = False
def load_data(dataset=None, mode='train'):
    
    # 定义数据迭代Batch大小
    BATCHSIZE = 256

    data_length = len(dataset)
    index_list = list(range(data_length))
    # 定义数据迭代加载器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        # 声明每个特征的列表
        usr_id_list,usr_gender_list,usr_age_list,usr_job_list = [], [], [], []
        mov_id_list,mov_tit_list,mov_cat_list,mov_poster_list = [], [], [], []
        score_list = []
        # 索引遍历输入数据集
        for idx, i in enumerate(index_list):
            # 获得特征数据保存到对应特征列表中
            usr_id_list.append(dataset[i]['usr_info']['usr_id'])
            usr_gender_list.append(dataset[i]['usr_info']['gender'])
            usr_age_list.append(dataset[i]['usr_info']['age'])
            usr_job_list.append(dataset[i]['usr_info']['job'])

            mov_id_list.append(dataset[i]['mov_info']['mov_id'])
            mov_tit_list.append(dataset[i]['mov_info']['title'])
            mov_cat_list.append(dataset[i]['mov_info']['category'])
            mov_id = dataset[i]['mov_info']['mov_id']

            if use_poster:
                # 不使用图像特征时，不读取图像数据，加快数据读取速度
                poster = Image.open(poster_path+'mov_id{}.jpg'.format(str(mov_id)))
                poster = poster.resize([64, 64])
                if len(poster.size) <= 2:
                    poster = poster.convert("RGB")

                mov_poster_list.append(np.array(poster))

            score_list.append(int(dataset[i]['scores']))
            # 如果读取的数据量达到当前的batch大小，就返回当前批次
            if len(usr_id_list)==BATCHSIZE:
                # 转换列表数据为数组形式，reshape到固定形状
                usr_id_arr = np.array(usr_id_list)
                usr_gender_arr = np.array(usr_gender_list)
                usr_age_arr = np.array(usr_age_list)
                usr_job_arr = np.array(usr_job_list)

                mov_id_arr = np.array(mov_id_list)

                mov_cat_arr = np.reshape(np.array(mov_cat_list), [BATCHSIZE, 6]).astype(np.int64)
                mov_tit_arr = np.reshape(np.array(mov_tit_list), [BATCHSIZE, 1, 15]).astype(np.int64)

                if use_poster:
                    mov_poster_arr = np.reshape(np.array(mov_poster_list)/127.5 - 1, [BATCHSIZE, 3, 64, 64]).astype(np.float32)
                else:
                    mov_poster_arr = np.array([0.])
                    
                scores_arr = np.reshape(np.array(score_list), [-1, 1]).astype(np.float32)
                
                # 返回当前批次数据
                yield [usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr], \
                       [mov_id_arr, mov_cat_arr, mov_tit_arr, mov_poster_arr], scores_arr
                
                # 清空数据
                usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
                mov_id_list, mov_tit_list, mov_cat_list, score_list = [], [], [], []
                mov_poster_list = []
    return data_generator

dataset = get_dataset(usr_info, rating_info, movie_info)
print("数据集总数量：", len(dataset))

trainset = dataset[:int(0.8*len(dataset))]
train_loader = load_data(trainset, mode="train")
print("训练集数量：", len(trainset))

validset = dataset[int(0.8*len(dataset)):]
valid_loader = load_data(validset, mode='valid')
print("验证集数量:", len(validset))

for idx, data in enumerate(train_loader()):
    usr_data, mov_data, score = data
    
    usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr = usr_data
    mov_id_arr, mov_cat_arr, mov_tit_arr, mov_poster_arr = mov_data
    print("用户ID数据尺寸", usr_id_arr.shape)
    print("电影ID数据尺寸", mov_id_arr.shape, ", 电影类别genres数据的尺寸", mov_cat_arr.shape, ", 电影名字title的尺寸", mov_tit_arr.shape)
    break
