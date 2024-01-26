import paddle
import paddle.nn as nn
import numpy as np

#todo 导入train_loader，valid_loader，valid_pm
from data_utils.data_loader import data_loader
from data_utils.valid_loader import valid_data_loader
from valider.valid_pm import valid_pm

def train_pm(model,
             datadir,
             annotiondir,
             optimizer,
             batch_size=10,
             EPOCH_NUM=20,
             use_gpu=False,
             save=None):
    # 使用0号GPU训练
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('********start training********')
    # 定义训练数据读取器train_loader和验证数据读取器valid_loader
    train_loader = data_loader(datadir=datadir + '/train_data/PALM-Training400', batch_size=batch_size, mode='train')
    valid_loader = valid_data_loader(datadir + '/PALM-Validation400', annotiondir)
    # 初始化模型对应参数的验证正确率
    model.max_accuracy, _ = valid_pm(model, valid_loader, batch_size=50)
    print('Initial max accuracy ：', model.max_accuracy)

    for epoch in range(EPOCH_NUM):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data).astype('int64')
            # 使用模型进行前向计算，得到预测值
            out = model(img)
            # 计算相应的损失值，并且得到对应的平均损失
            loss = nn.functional.cross_entropy(out, label, reduction='none')
            avg_loss = paddle.mean(loss)

            if batch_id % 10 == 0:  # 每10个batch输出1次训练结果
                print("epoch:{}===batch_id:{}===loss:{:.4f}".format(
                    epoch, batch_id, float(avg_loss.numpy())))

            # 反向传播，更新权重，消除梯度
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch进行一次训练集的验证，获取模型在验证集上的正确率和损失值
        valid_accuracy, valid_loss = valid_pm(model, valid_loader, batch_size=50)
        print('[validation]:======accuracy:{:.5f}/loss:{:.5f}'.format(valid_accuracy, valid_loss))

        # 如果模型准确率上升并且存在一个模型保存的策略，那么保存模型
        if save != None and valid_accuracy > model.max_accuracy:
            save(valid_accuracy, model)
            print('max accuracy :', model.max_accuracy)
        print()
    print('Final max accuracy :', model.max_accuracy)


