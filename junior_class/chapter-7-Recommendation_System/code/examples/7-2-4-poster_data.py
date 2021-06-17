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


# %matplotlib inline
from PIL import Image
import matplotlib.pyplot as plt

# 使用海报图像和不使用海报图像的文件路径不同，处理方式相同
use_poster = True
if use_poster:
    rating_path = "./work/ml-1m/new_rating.txt"
else:
    rating_path = "./work/ml-1m/ratings.dat"
    
with open(rating_path, 'r') as f:
    data = f.readlines()
    
# 从新的rating文件中收集所有的电影ID
mov_id_collect = []
for item in data:
    item = item.strip().split("::")
    usr_id,movie_id,score = item[0],item[1],item[2]
    mov_id_collect.append(movie_id)

# 根据电影ID读取图像
poster_path = "./work/ml-1m/posters/"

# 显示mov_id_collect中第几个电影ID的图像
idx = 1

poster = Image.open(poster_path+'mov_id{}.jpg'.format(str(mov_id_collect[idx])))
# poster = poster.resize([64, 64])
plt.figure("Image") # 图像窗口名称
plt.imshow(poster)
plt.axis('on') # 关掉坐标轴为 off
plt.title("poster with ID {}".format(mov_id_collect[idx])) # 图像题目
plt.show()