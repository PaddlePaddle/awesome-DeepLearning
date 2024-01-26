#导入相关的库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random
import matplotlib.pyplot as plt



# 编写加载数据的函数
def load_data():
    datafile='./data/housing.data'
    data=np.fromfile(datafile,sep=' ',dtype=np.float32)
    #读取数据文件
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    #将一维的数据转成二维
    ratio=0.8
    #设置训练集的比例为0.8
    offset=int(data.shape[0]*ratio)
    training_data=data[:offset]
    #对数据进行标准化
    maximums,minimums,avgs=training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0)/training_data.shape[0]
    global max_values
    global min_values
    global avg_values
    max_values=maximums
    min_values=minimums
    avg_values=avgs
    for i in range(feature_num):
        data[:,i]=(data[:,i]-avgs[i])/(maximums[i]-minimums[i])
    training_data=data[:offset]
    test_data=data[offset:]
    return training_data,test_data



# 定义网络结构为两层，全连接层+激活函数+全连接层
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net,self).__init__()

        self.fc1=Linear(in_features=13,out_features=128)
        self.fc2=Linear(in_features=128,out_features=1)
    def forward(self,inputs):
        x=self.fc1(inputs)
        x=F.relu(x)
        x=self.fc2(x)
        return x

#创建网络的一个对象
model=Net()
model.train()
#进入模型的训练模式
training_data, test_data = load_data()
opt = paddle.optimizer.Adadelta(learning_rate=0.01, parameters=model.parameters())
#设置优化器为Adadelta


EPOCH_NUM = 100   #设置迭代次数
BATCH_SIZE = 10  #设置批次大小

# 定义外层循环
losses=[]
for epoch_id in range(1,EPOCH_NUM+1):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1]) # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:]) # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 前向计算
        predicts = model(house_features)

        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        losses.append(avg_loss.numpy())
        if iter_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()



plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()



def load_one_example():
    # 从上边已加载的测试集中，随机选择一条作为测试数据
    idx = np.random.randint(0, test_data.shape[0])
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 修改该条数据shape为[1,13]
    one_data =  one_data.reshape([1,-1])

    return one_data, label



# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功，模型参数保存在LR_model.pdparams中")


# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
one_data, label = load_one_example()
# 将数据转为动态图的variable格式 
one_data = paddle.to_tensor(one_data)
predict = model(one_data)

# 对结果做反归一化处理
predict = predict * (max_values[-1] - min_values[-1]) + avg_values[-1]
# 对label数据做反归一化处理
label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]

print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))
