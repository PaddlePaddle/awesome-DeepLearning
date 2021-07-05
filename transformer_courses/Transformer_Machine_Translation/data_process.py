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


from tqdm import tqdm
import os
import xml.etree.ElementTree as ET




def filter_out_html(filename1,filename2):
	f1 = open(filename1,'r')
	f2 = open(filename2,'r')

	data1 = f1.readlines()
	data2 = f2.readlines()
	assert len(data1)==len(data2)#用codecs会导致报错不知道为什么
	fw1 = open(filename1+".txt",'w')
	fw2 = open(filename2+".txt",'w')

	for line1,line2 in tqdm(zip(data1,data2)):
		line1 = line1.strip()
		line2 = line2.strip()
		if line1 and line2:
			if '<' not in line1 and '>' not in line1 and '<' not in line2 and '>' not in line2:
				fw1.write(line1+"\n")
				fw2.write(line2+"\n")
	fw1.close()
	f1.close()
	fw2.close()
	f2.close()

	return filename1+".txt",filename2+".txt"

en_dir='zh-en/train.tags.zh-en.en'
zn_dir='zh-en/train.tags.zh-en.zh'
filter_out_html(en_dir,zn_dir)
tree_source_dev = ET.parse('zh-en/IWSLT15.TED.dev2010.zh-en.zh.xml')
tree_source_dev = [seg.text for seg in tree_source_dev.iter('seg')]

tree_target_dev = ET.parse('zh-en/IWSLT15.TED.dev2010.zh-en.en.xml')
tree_target_dev = [seg.text for seg in tree_target_dev.iter('seg')]

with open('dev_cn.txt','w') as f:
    for item in tree_source_dev:
        f.write(item+'\n')

with open('dev_en.txt','w') as f:
    for item in tree_target_dev:
        f.write(item+'\n')

tree_source_test = ET.parse('zh-en/IWSLT15.TED.tst2011.zh-en.zh.xml')
tree_source_test = [seg.text for seg in tree_source_test.iter('seg')]

tree_target_test = ET.parse('zh-en/IWSLT15.TED.tst2011.zh-en.en.xml')
tree_target_test = [seg.text for seg in tree_target_test.iter('seg')]

with open('test_cn.txt','w') as f:
    for item in tree_source_test:
        f.write(item+'\n')

with open('test_en.txt','w') as f:
    for item in tree_target_test:
        f.write(item+'\n')



