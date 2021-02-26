
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

import random
import numpy as np
from PIL import Image

class MovieLen(object):
    def __init__(self, use_poster):
        self.use_poster = use_poster
        # 声明每个数据文件的路径
        usr_info_path = "./work/ml-1m/users.dat"
        if use_poster:
            rating_path = "./work/ml-1m/new_rating.txt"
        else:
            rating_path = "./work/ml-1m/ratings.dat"

        movie_info_path = "./work/ml-1m/movies.dat"
        self.poster_path = "./work/ml-1m/posters/"
        # 得到电影数据
        self.movie_info, self.movie_cat, self.movie_title = self.get_movie_info(movie_info_path)
        # 记录电影的最大ID
        self.max_mov_cat = np.max([self.movie_cat[k] for k in self.movie_cat])
        self.max_mov_tit = np.max([self.movie_title[k] for k in self.movie_title])
        self.max_mov_id = np.max(list(map(int, self.movie_info.keys())))
        # 记录用户数据的最大ID
        self.max_usr_id = 0
        self.max_usr_age = 0
        self.max_usr_job = 0
        # 得到用户数据
        self.usr_info = self.get_usr_info(usr_info_path)
        # 得到评分数据
        self.rating_info = self.get_rating_info(rating_path)
        # 构建数据集 
        self.dataset = self.get_dataset(usr_info=self.usr_info,
                                        rating_info=self.rating_info,
                                        movie_info=self.movie_info)
        # 划分数据集，获得数据加载器
        self.train_dataset = self.dataset[:int(len(self.dataset)*0.9)]
        self.valid_dataset = self.dataset[int(len(self.dataset)*0.9):]
        print("##Total dataset instances: ", len(self.dataset))
        print("##MovieLens dataset information: \nusr num: {}\n"
              "movies num: {}".format(len(self.usr_info),len(self.movie_info)))
    # 得到电影数据
    def get_movie_info(self, path):
        # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中 
        with open(path, 'r', encoding="ISO-8859-1") as f:
            data = f.readlines()
        # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
        movie_info, movie_titles, movie_cat = {}, {}, {}
        # 对电影名字、类别中不同的单词计数
        t_count, c_count = 1, 1

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

    def get_usr_info(self, path):
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
            self.max_usr_id = max(self.max_usr_id, int(usr_id))
            self.max_usr_age = max(self.max_usr_age, int(item[2]))
            self.max_usr_job = max(self.max_usr_job, int(item[3]))
        return use_info
    # 得到评分数据
    def get_rating_info(self, path):
        # 读取文件里的数据
        with open(path, 'r') as f:
            data = f.readlines()
        # 将数据保存在字典中并返回
        rating_info = {}
        for item in data:
            item = item.strip().split("::")
            usr_id,movie_id,score = item[0],item[1],item[2]
            if usr_id not in rating_info.keys():
                rating_info[usr_id] = {movie_id:float(score)}
            else:
                rating_info[usr_id][movie_id] = float(score)
        return rating_info
    # 构建数据集
    def get_dataset(self, usr_info, rating_info, movie_info):
        trainset = []
        for usr_id in rating_info.keys():
            usr_ratings = rating_info[usr_id]
            for movie_id in usr_ratings:
                trainset.append({'usr_info': usr_info[usr_id],
                                 'mov_info': movie_info[movie_id],
                                 'scores': usr_ratings[movie_id]})
        return trainset
    
    def load_data(self, dataset=None, mode='train'):
        use_poster = False

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
                    poster = Image.open(self.poster_path+'mov_id{}.jpg'.format(str(mov_id[0])))
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

                    # 放回当前批次数据
                    yield [usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr], \
                           [mov_id_arr, mov_cat_arr, mov_tit_arr, mov_poster_arr], scores_arr

                    # 清空数据
                    usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
                    mov_id_list, mov_tit_list, mov_cat_list, score_list = [], [], [], []
                    mov_poster_list = []
        return data_generator

# 声明数据读取类
dataset = MovieLen(False)
# 定义数据读取器
train_loader = dataset.load_data(dataset=dataset.train_dataset, mode='train')
# 迭代的读取数据， Batchsize = 256
for idx, data in enumerate(train_loader()):
    usr, mov, score = data
    print("打印用户ID，性别，年龄，职业数据的维度：")
    for v in usr:
        print(v.shape)
    print("打印电影ID，名字，类别数据的维度：")
    for v in mov:
        print(v.shape)
    
    break