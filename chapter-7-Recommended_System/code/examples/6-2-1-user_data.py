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


# 解压数据集
# unzip -o -q -d ~/work/ ~/data/data19736/ml-1m.zip

import numpy as np
usr_file = "./work/ml-1m/users.dat"
# 打开文件，读取所有行到data中
with open(usr_file, 'r') as f:
    data = f.readlines()
# 打印data的数据长度、第一条数据、数据类型
print("data 数据长度是：",len(data))
print("第一条数据是：", data[0])
print("数据类型：", type(data[0]))


def gender2num(gender):
    return 1 if gender == 'F' else 0
print("性别M用数字 {} 表示".format(gender2num('M')))
print("性别F用数字 {} 表示".format(gender2num('F')))


usr_info = {}
max_usr_id = 0
#按行索引数据
for item in data:
    # 去除每一行中和数据无关的部分
    item = item.strip().split("::")
    usr_id = item[0]
    # 将字符数据转成数字并保存在字典中
    usr_info[usr_id] = {'usr_id': int(usr_id),
                        'gender': gender2num(item[1]),
                        'age': int(item[2]),
                        'job': int(item[3])}
    max_usr_id = max(max_usr_id, int(usr_id))

print("用户ID为3的用户数据是：", usr_info['3'])

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
print("用户数量:", len(usr_info))
print("最大用户ID:", max_usr_id)
print("第1个用户的信息是：", usr_info['1'])


