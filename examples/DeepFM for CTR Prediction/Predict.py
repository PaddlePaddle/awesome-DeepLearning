import paddle

testFileLineNum = 1788219
testFilePath = './work/slot_test_data_full'
batchsize = 400
testBatchNum = testFileLineNum // batchsize
deepFM_TestDataset = DeepFM_Dataset(batchsize, testFilePath)


def predict(deepFM_model, deepFM_Dataset, batchnum):
    for batchidx in range(batchnum):
        # 加载数据
        data = deepFM_Dataset.getNextBatchData()
        label_data = paddle.to_tensor(data[0], dtype='float32')  # [batchsize,]
        label_data = paddle.unsqueeze(label_data, axis=1)  # [batchsize,1]
        # 得到特征
        sparse_feature = paddle.to_tensor(data[1], dtype='int64')
        dense_feature = paddle.to_tensor(data[2], dtype='float32')
        # 得到预测值，为了得到每条样本分属于正负样本的概率，将预测结果和1-predict合并起来得到predicts，以便接下来计算auc
        predicts1 = deepFM_model(sparse_feature, dense_feature)  # [batchsize,1]
        predicts0 = 1 - predicts1  # [batchsize,1]
        predicts = paddle.concat([predicts0, predicts1], axis=1)
        # 计算auc
        auc = paddle.metric.Auc()
        auc.update(preds=predicts, labels=label_data)
        loss = F.binary_cross_entropy(predicts1, label_data)

        if batchidx % (batchnum // 20) == 0:
            print(paddle.concat([predicts[:4, ], label_data[:4, ]], axis=1).numpy())
            print("batchidx:{} loss:{} auc:{}".format(batchidx, loss.numpy(), auc.accumulate()))


predict(testDeepFM_model, deepFM_TestDataset, testBatchNum)