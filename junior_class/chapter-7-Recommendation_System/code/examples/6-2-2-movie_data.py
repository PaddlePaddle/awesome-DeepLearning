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

movie_info_path = "./work/ml-1m/movies.dat"
# 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
with open(movie_info_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()

# 读取第一条数据，并打印
item = data[0]
print(item)
item = item.strip().split("::")
print("movie ID:", item[0])
print("movie title:", item[1][:-7])
print("movie year:", item[1][-5:-1])
print("movie genre:", item[2].split('|'))


movie_info_path = "./work/ml-1m/movies.dat"
# 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
with open(movie_info_path, 'r', encoding="ISO-8859-1") as f:
    data = f.readlines()
    
movie_info = {}
for item in data:
    item = item.strip().split("::")
    # 获得电影的ID信息
    v_id = item[0]
    movie_info[v_id] = {'mov_id': int(v_id)}
max_id = max([movie_info[k]['mov_id'] for k in movie_info.keys()])
print("电影的最大ID是：", max_id)

# 用于记录电影title每个单词对应哪个序号
movie_titles = {}
#记录电影名字包含的单词最大数量
max_title_length = 0
# 对不同的单词从1 开始计数
t_count = 1
# 按行读取数据并处理
for item in data:
    item = item.strip().split("::")
    # 1. 获得电影的ID信息
    v_id = item[0]
    v_title = item[1][:-7] # 去掉title中年份数据
    v_year = item[1][-5:-1]
    titles = v_title.split()
    # 获得title最大长度
    max_title_length = max((max_title_length, len(titles)))
    
    # 2. 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
    for t in titles:
        if t not in movie_titles:
            movie_titles[t] = t_count
            t_count += 1
            
    v_tit = [movie_titles[k] for k in titles]
    # 保存电影ID数据和title数据到字典中
    movie_info[v_id] = {'mov_id': int(v_id),
                        'title': v_tit,
                        'years': int(v_year)}
    
print("最大电影title长度是：",  max_title_length)
ID = 1
# 读取第一条数据，并打印
item = data[0]
item = item.strip().split("::")
print("电影 ID:", item[0])
print("电影 title:", item[1][:-7])
print("ID为1 的电影数据是：", movie_info['1'])


# 用于记录电影类别每个单词对应哪个序号
movie_titles, movie_cat = {}, {}
max_title_length = 0
max_cat_length = 0

t_count, c_count = 1, 1
# 按行读取数据并处理
for item in data:
    item = item.strip().split("::")
    # 1. 获得电影的ID信息
    v_id = item[0]
    cats = item[2].split('|')

    # 获得电影类别数量的最大长度
    max_cat_length = max((max_cat_length, len(cats)))
            
    v_cat = item[2].split('|')
    # 3. 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
    for cat in cats:
        if cat not in movie_cat:
            movie_cat[cat] = c_count
            c_count += 1
    v_cat = [movie_cat[k] for k in v_cat]
    
    # 保存电影ID数据和title数据到字典中
    movie_info[v_id] = {'mov_id': int(v_id),
                        'category': v_cat}
    
print("电影类别数量最多是：",  max_cat_length)
ID = 1
# 读取第一条数据，并打印
item = data[0]
item = item.strip().split("::")
print("电影 ID:", item[0])
print("电影种类 category:", item[2].split('|'))
print("ID为1 的电影数据是：", movie_info['1'])

# 建立三个字典，分别存放电影ID、名字和类别
movie_info, movie_titles, movie_cat = {}, {}, {}
# 对电影名字、类别中不同的单词从 1 开始标号
t_count, c_count = 1, 1

count_tit = {}
# 按行读取数据并处理
for item in data:
    item = item.strip().split("::")
    # 1. 获得电影的ID信息
    v_id = item[0]
    v_title = item[1][:-7] # 去掉title中年份数据
    cats = item[2].split('|')
    v_year = item[1][-5:-1]

    titles = v_title.split()
    # 2. 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
    for t in titles:
        if t not in movie_titles:
            movie_titles[t] = t_count
            t_count += 1
    # 3. 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
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
    # 4. 保存电影数据到movie_info中
    movie_info[v_id] = {'mov_id': int(v_id),
                        'title': v_tit,
                        'category': v_cat,
                        'years': int(v_year)}
    
print("电影数据数量：", len(movie_info))
ID = 2
print("原始的电影ID为 {} 的数据是：".format(ID), data[ID-1])
print("电影ID为 {} 的转换后数据是：".format(ID), movie_info[str(ID)])

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
print("电影数量：", len(movie_info))
ID = 1
print("原始的电影ID为 {} 的数据是：".format(ID), data[ID-1])
print("电影ID为 {} 的转换后数据是：".format(ID), movie_info[str(ID)])

print("电影种类对应序号：'Animation':{} 'Children's':{} 'Comedy':{}".format(movie_cat['Animation'], 
                                                                   movie_cat["Children's"], 
                                                                   movie_cat['Comedy']))
print("电影名称对应序号：'The':{} 'Story':{} ".format(movie_titles['The'], movie_titles['Story']))

