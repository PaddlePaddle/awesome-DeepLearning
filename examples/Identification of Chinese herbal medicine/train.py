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
import utils
import paddle
from dataloader import dataset
from PIL import Image
from model import VGGNet
import matplotlib.pyplot as plt


# 折线图，用于观察训练过程中loss和acc的走势
def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    train_parameters = {
        "src_path":
        "/home/aistudio/data/data105575/Chinese Medicine.zip",  #原始数据集路径
        "target_path": "/home/aistudio/data/",  #要解压的路径
        "train_list_path": "/home/aistudio/data/train.txt",  #train.txt路径
        "eval_list_path": "/home/aistudio/data/eval.txt",  #eval.txt路径
        "label_dict": {},  #标签字典
        "readme_path": "/home/aistudio/data/readme.json",  #readme.json路径
        "class_dim": -1,  #分类数
        "input_size": [3, 224, 224],  #输入图片的shape
        "num_epochs": 35,  #训练轮数
        "skip_steps": 10,  #训练时输出日志的间隔
        "save_steps": 100,  #训练时保存模型参数的间隔
        "learning_strategy": {  #优化函数相关的配置
            "lr": 0.0001  #超参数学习率
        },
        "checkpoints": "/home/aistudio/work/checkpoints"  #保存的路径
    }

    src_path = train_parameters['src_path']
    target_path = train_parameters['target_path']
    train_list_path = train_parameters['train_list_path']
    eval_list_path = train_parameters['eval_list_path']

    # 调用解压函数解压数据集
    utils.unzip_data(src_path, target_path)

    # 划分训练集与验证集，乱序，生成数据列表
    #每次生成数据列表前，首先清空train.txt和eval.txt
    with open(train_list_path, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(eval_list_path, 'w') as f:
        f.seek(0)
        f.truncate()
    # 生成数据列表
    train_parameters = utils.get_data_list(target_path, train_list_path,
                                           eval_list_path, train_parameters)

    #训练数据加载
    train_dataset = dataset('/home/aistudio/data', mode='train')
    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    model = VGGNet(train_parameters['class_dim'])
    model.train()
    # 配置loss函数
    cross_entropy = paddle.nn.CrossEntropyLoss()
    # 配置参数优化器
    optimizer = paddle.optimizer.Adam(
        learning_rate=train_parameters['learning_strategy']['lr'],
        parameters=model.parameters())
    #保存类别字典用于测试输出
    with open('/home/aistudio/label_dict.txt', 'w') as fd:
        for key, v in train_parameters['label_dict'].items():
            fd.write(key + ' ' + v + '\n')

    steps = 0
    Iters, total_loss, total_acc = [], [], []

    for epo in range(train_parameters['num_epochs']):
        for _, data in enumerate(train_loader()):
            steps += 1
            x_data = data[0]
            y_data = data[1]
            predicts, acc = model(x_data, y_data)
            loss = cross_entropy(predicts, y_data)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if steps % train_parameters["skip_steps"] == 0:
                Iters.append(steps)
                total_loss.append(loss.numpy()[0])
                total_acc.append(acc.numpy()[0])
                # 打印中间过程
                print('epo: {}, step: {}, loss is: {}, acc is: {}'\
                      .format(epo, steps, loss.numpy(), acc.numpy()))
            #保存模型参数
            if steps % train_parameters["save_steps"] == 0:
                save_path = train_parameters[
                    "checkpoints"] + "/" + "save_dir_" + str(
                        steps) + '.pdparams'
                print('save model to: ' + save_path)
                paddle.save(model.state_dict(), save_path)
    paddle.save(
        model.state_dict(),
        train_parameters["checkpoints"] + "/" + "save_dir_final.pdparams")
    #打印训练过程的loss和scc曲线,如果需要请自行打开
    #draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
    #draw_process("trainning acc","green",Iters,total_acc,"trainning acc")
