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

import jieba

# 中文Jieba分词
def jieba_cut(in_file,out_file):
    out_f = open(out_file,'w',encoding='utf8')
    with open(in_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            cut_line = ' '.join(jieba.cut(line))
            out_f.write(cut_line+'\n')
    out_f.close()


zn_dir='zh-en/train.tags.zh-en.zh.txt'
cut_zn_dir='zh-en/train.tags.zh-en.zh.cut.txt'
jieba_cut(zn_dir,cut_zn_dir)

zn_dir='dev_cn.txt'
cut_zn_dir='dev_cn.cut.txt'
jieba_cut(zn_dir,cut_zn_dir)

zn_dir='dev_cn.txt'
cut_zn_dir='dev_cn.cut.txt'
jieba_cut(zn_dir,cut_zn_dir)

zn_dir='test_cn.txt'
cut_zn_dir='test_cn.cut.txt'
jieba_cut(zn_dir,cut_zn_dir)









