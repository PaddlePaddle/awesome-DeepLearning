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

special_token=['<s>','<e>','<unk>']
cn_vocab=[]
with open('zh-en/temp1') as f:
    for item in f.readlines():
        words=item.strip().split()
        cn_vocab.append(words[0])

with open('zh-en/vocab.ch.src','w') as f:
    for item in special_token:
        f.write(item+'\n')
    for item in cn_vocab:
        f.write(item+'\n')

eng_vocab=[]
with open('zh-en/temp2') as f:
    for item in f.readlines():
        words=item.strip().split()
        eng_vocab.append(words[0])

with open('zh-en/vocab.en.tgt','w') as f:
    for item in special_token:
        f.write(item+'\n')
    for item in eng_vocab:
        f.write(item+'\n')
