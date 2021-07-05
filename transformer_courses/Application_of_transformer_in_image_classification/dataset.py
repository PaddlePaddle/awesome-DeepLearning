# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from paddle.io import Dataset
from transform import  transform

# 读取数据，如果是训练数据，随即打乱数据顺序
def get_file_list(file_list):
    with open(file_list) as flist:
        full_lines = [line.strip() for line in flist]

    return full_lines


# 定义数据读取器
class ImageNetDataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.full_lines = get_file_list(file_list)
        self.delimiter = ' '
        self.num_samples = len(self.full_lines)
        self.data_dir = data_dir
        return

    def __getitem__(self, idx):
        line = self.full_lines[idx]
        img_path, label = line.split(self.delimiter)
        img_path = os.path.join(self.data_dir, img_path)
        with open(img_path, 'rb') as f:
            img = f.read()

        transformed_img = transform(img)
        return (transformed_img, int(label))

    def __len__(self):
        return self.num_samples