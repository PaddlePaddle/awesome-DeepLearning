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

import paddle
from paddle.nn import Linear, Embedding, Conv2D
import numpy as np
import paddle.nn.functional as F

from math import sqrt
from nets.DSSM import Model
from movielens_dataset import MovieLen

def train(model,train_loader,Epoches):
    # 配置训练参数
    lr = 0.001
    paddle.set_device('cpu') 

    # 启动训练
    model.train()
    # 获得数据读取器
    data_loader = train_loader
    # 使用adam优化器，学习率使用0.01
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    
    for epoch in range(0, Epoches):
        for idx, data in enumerate(data_loader()):
            # 获得数据，并转为tensor格式
            usr, mov, score = data
            usr_v = [paddle.to_tensor(var) for var in usr]
            mov_v = [paddle.to_tensor(var) for var in mov]
            scores_label = paddle.to_tensor(score)
            # 计算出算法的前向计算结果
            _, _, scores_predict = model(usr_v, mov_v)
            # 计算loss
            loss = F.square_error_cost(scores_predict, scores_label)
            avg_loss = paddle.mean(loss)

            if idx % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, avg_loss.numpy()))
                
            # 损失函数下降，并清除梯度
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        # 每个epoch 保存一次模型
        paddle.save(model.state_dict(), './checkpoint/epoch'+str(epoch)+'.pdparams')


def evaluation(model, params_file_path,valid_loader):
    print(params_file_path)
    # print(model.parameters())
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    acc_set = []
    avg_loss_set = []
    squaredError=[]
    for idx, data in enumerate(valid_loader()):
        usr, mov, score_label = data
        usr_v = [paddle.to_tensor(var) for var in usr]
        mov_v = [paddle.to_tensor(var) for var in mov]
        _, _, scores_predict = model(usr_v, mov_v)

        pred_scores = scores_predict.numpy()
        # print(usr_v)
        # print(mov_v)
        # print(pred_scores)
        avg_loss_set.append(np.mean(np.abs(pred_scores - score_label)))
        squaredError.extend(np.abs(pred_scores - score_label)**2)

        diff = np.abs(pred_scores - score_label)
        diff[diff>0.5] = 1
        acc = 1 - np.mean(diff)
        acc_set.append(acc)
        break
    RMSE=sqrt(np.sum(squaredError) / len(squaredError))
    return np.mean(acc_set), np.mean(avg_loss_set),RMSE


if __name__=="__main__":

    # 启动训练
    fc_sizes=[128, 64, 32]
    Epoches=2
    # 定义数据迭代Batch大小
    BATCHSIZE = 256
    use_poster, use_mov_title, use_mov_cat, use_age_job = False, True, True, True
    data_path='../data/'
    Dataset = MovieLen(use_poster,data_path)
    trainset = Dataset.train_dataset
    valset = Dataset.valid_dataset
    train_loader = Dataset.load_data(dataset=trainset, mode='train',batch_size=BATCHSIZE)
    valid_loader =Dataset.load_data(dataset=valset, mode='valid',batch_size=BATCHSIZE)
    model = Model(use_poster, use_mov_title, use_mov_cat, use_age_job,fc_sizes,Dataset)
    # train(model,train_loader,Epoches)

    param_path = "./checkpoint/epoch"
    for i in range(Epoches):
        acc, mae,rmse = evaluation(model, param_path+str(i)+'.pdparams',valid_loader)
        print("ACC:", acc, "MAE:", mae,'RMSE:',rmse)
