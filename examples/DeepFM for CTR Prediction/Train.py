import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np

#模型参数
sparse_feature_number = 1000001 # 1000001 离散特征数
embedding_dim = 9# 9 嵌入层维度
dense_feature_dim = 13#13 稠密特征维度
sparse_num_field = 26# sparse_inputs_slots-1==>26 稀疏特征维度
layer_sizes = [512, 256, 128, 32]#  fc_sizes: [512, 256, 128, 32] 隐藏层数量

#训练参数
epochs = 2
batchsize=50
learning_rate=1e-3


def train(
        deepFM_model,
        deepFM_Dataset,
        batchnum,
        optimizer,
        sparse_feature_number=1000001,
        embedding_dim=9,
        dense_feature_dim=13,
        sparse_num_field=26,
        layer_sizes=[512, 256, 128, 32],

        epochs=1,
        batchsize=500,
        learning_rate=1e-3):
    lossFunc = F.binary_cross_entropy
    for epoch in range(epochs):
        for batchidx in range(batchnum):
            # 加载训练数据
            data = deepFM_Dataset.getNextBatchData()
            label_data = paddle.to_tensor(data[0], dtype='float32')  # [batchsize,]
            label_data = paddle.unsqueeze(label_data, axis=1)  # [batchsize,1]
            # 得到稀疏/稠密特征
            sparse_feature = paddle.to_tensor(data[1], dtype='int64')
            dense_feature = paddle.to_tensor(data[2], dtype='float32')
            # 得到预测值，为了得到每条样本分属于正负样本的概率，将预测结果和1-predict合并起来得到predicts，以便接下来计算auc
            predicts1 = deepFM_model(sparse_feature, dense_feature)  # [batchsize,1]
            predicts0 = 1 - predicts1  # [batchsize,1]
            predicts = paddle.concat([predicts0, predicts1], axis=1)
            # 计算auc指标
            auc = paddle.metric.Auc()
            auc.update(preds=predicts, labels=label_data)
            loss = lossFunc(predicts1, label_data)
            loss.backward()
            if batchidx % (batchnum // 220) == 0:
                print("processing:{}%".format(100 * batchidx / batchnum))
                print("label data 0-num: {0}  1-num:{1}".format(np.sum(data[0] < 0.5), np.sum(data[0] > 0.5)))
                print("epoch: {}, batch_id: {}, loss : {}, auc: {}".format(epoch, batchidx, loss.numpy(),
                                                                           auc.accumulate()))

            adam.step()
            adam.clear_grad()


epochs = 1
learning_rate = 1e-3
trainFilePath = './work/slot_train_data_full'
trainFilesLineNum = 40000000
# trainFilesLineNum=200000
batchsize = 2000
trainBatchNum = trainFilesLineNum // batchsize

deepFM_TrainDataset = DeepFM_Dataset(batchsize, trainFilePath)
deepFM_model = DeepFMLayer(sparse_feature_number, embedding_dim, dense_feature_dim, sparse_num_field, layer_sizes)
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=deepFM_model.parameters())  # Adam优化器

train(deepFM_model, deepFM_TrainDataset, epochs=epochs, batchsize=batchsize, batchnum=trainBatchNum,
      learning_rate=learning_rate, optimizer=adam)