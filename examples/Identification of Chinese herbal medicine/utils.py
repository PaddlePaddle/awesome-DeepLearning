# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# coding=utf-8
import os
import json
import random
import zipfile


def unzip_data(src_path, target_path):
    if (not os.path.isdir(target_path + "Chinese Medicine")):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def get_data_list(target_path, train_list_path, eval_list_path,
                  train_parameters):
    '''
    生成数据列表
    '''
    #存放所有类别的信息
    class_detail = []
    #获取所有类别保存的文件夹名称
    data_list_path = target_path + "Chinese Medicine/"
    class_dirs = os.listdir(data_list_path)
    #总的图像数量
    all_class_images = 0
    #存放类别标签
    class_label = 0
    #存放类别数目
    class_dim = 0
    #存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    #读取每个类别，['baihe', 'gouqi','jinyinhua','huaihua','dangshen']
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            #每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            #统计每个类别有多少张图片
            class_sum = 0
            #获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + '/' + img_path  # 每张图片的路径
                if class_sum % 8 == 0:  # 每8张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label +
                                        "\n")  #trainer_sum测试数据的数目
                class_sum += 1  #每类图片的数目
                all_class_images += 1  #所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir  #类别名称
            class_detail_list['class_label'] = class_label  #类别标签
            class_detail_list['class_eval_images'] = eval_sum  #该类数据的测试集数目
            class_detail_list[
                'class_trainer_images'] = trainer_sum  #该类数据的训练集数目
            class_detail.append(class_detail_list)
            #初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1

    #初始化分类数
    train_parameters['class_dim'] = class_dim

    #乱序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  #文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(
        readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')
    return train_parameters
