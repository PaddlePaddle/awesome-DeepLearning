import paddle
import paddle.nn as nn
import numpy as np


def valid_pm(model, valid_loader, batch_size=100):
    model.eval()
    print("*****valid data import success*****")
    batch_accuracy = []
    batch_loss = []
    for batch_id, data in enumerate(valid_loader(batch_size=batch_size)):
        # 加载数据，并且进行类型转换
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data).astype('int64')

        # 前向计算，计算预测值
        out = model(img)
        predict = paddle.argmax(out, 1)

        # 计算损失值和准确率，并且加入到相应列表中
        loss = nn.functional.cross_entropy(out, label, reduction='none')
        avg_loss = paddle.mean(loss)
        accuracy = sum(predict.numpy().reshape(-1, 1) == label.numpy()) / float(label.shape[0])
        batch_loss.append(float(avg_loss.numpy()))
        batch_accuracy.append(accuracy)

    # 将所有批次的损失值和准确率平均，得到最终损失值和准确率
    avg_loss = np.mean(batch_loss)
    avg_accuracy = np.mean(batch_accuracy)
    return avg_accuracy, avg_loss
