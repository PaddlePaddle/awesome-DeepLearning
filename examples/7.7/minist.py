from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np 
from paddle.metric import Accuracy
import random


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
#归一化

#读取训练集 测试集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')


class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=150)
        self.linear2 = paddle.nn.Linear(in_features=150, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)
    #前馈网络
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

model = paddle.Model(LeNet())   # 封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()) # adam优化器

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# 训练模型
model.fit(train_dataset,epochs=2,batch_size=64,verbose=1)
#评估
model.evaluate(test_dataset, batch_size=64, verbose=1)

#训练
def train(model,Batch_size=64):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    model.train()
    iterator = 0
    epochs = 10
    total_steps = (int(50000//Batch_size)+1)*epochs
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01,decay_steps=total_steps,end_lr=0.001)
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
                log_writer.add_scalar(tag='acc',step=iterator,value=acc.numpy())
                log_writer.add_scalar(tag='loss',step=iterator,value=loss.numpy())
                iterator+=200
            optim.step()
            optim.clear_grad()
        paddle.save(model.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdparams')
        paddle.save(optim.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdopt')


#测试
def test(model):
    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    model.eval()
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

#随机抽取100张图片进行测试
def random_test(model,num=100):
    select_id = random.sample(range(1, 10000), 100) #生成一百张测试图片的下标
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        label = data[1]
    predicts = model(x_data)
    #返回正确率
    acc = paddle.metric.accuracy(predicts, label)
    print("正确率为：{}".format(acc.numpy()))

    
if __name__ == '__main__':
    model = LeNet()
    train(model)
    test(model)
    random_test(model)