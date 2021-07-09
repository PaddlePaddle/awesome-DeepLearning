import json
import gzip
import numpy as np
import random
import paddle


# 数据准备
def load_data(mode='train'):
    datafile = './work/datasets/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    # 加载json数据文件
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')
   
    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data
    if mode=='train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode=='valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode=='eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    print("训练数据集数量: ", len(imgs))
    
    # 获得数据集长度
    imgs_length = len(imgs)

    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的批次大小
    BATCHSIZE = 100
    
    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            # 将数据处理成希望的类型
            img = np.reshape(imgs[i], [1, 28, 28]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                # 获得一个batchsize的数据，并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []
    
        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
    return data_generator


# 基于ResNet50实现MNIST任务
resnet50 = paddle.vision.models.resnet50(num_classes = 10)
resnet50.conv1 = paddle.nn.Conv2D(1, 64, kernel_size=7, stride=2, padding=3)


# 模型训练、评估与保存
import paddle.nn.functional as F

class Trainer(object):
    def __init__(self, model_path, model, optimizer):
        self.model_path = model_path   # 模型存放路径 
        self.model = model             # 定义的模型
        self.optimizer = optimizer     # 优化器

    def save(self):
        # 保存模型
        paddle.save(self.model.state_dict(), self.model_path)

    def val_epoch(self, datasets):
        self.model.eval()  # 将模型设置为评估状态
        acc = list()
        for batch_id, data in enumerate(datasets()):
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            pred = self.model(images)   # 获取预测值
            # 取 pred 中得分最高的索引作为分类结果
            pred = paddle.argmax(pred, axis=-1)  
            res = paddle.equal(pred, labels)
            res = paddle.cast(res, dtype='float32')
            acc.extend(res.numpy())  # 追加
        acc = np.array(acc).mean()
        return acc

    def train_step(self, data):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        # 前向计算的过程
        predicts = self.model(images)
        # 计算损失
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        # 后向传播，更新参数的过程
        avg_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return avg_loss

    def train_epoch(self, datasets, epoch):
        self.model.train()
        for batch_id, data in enumerate(datasets()):
            loss = self.train_step(data)
            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

    def train(self, train_datasets, val_datasets, epochs):
        for i in range(epochs):
            self.train_epoch(train_datasets, i)
            train_acc = self.val_epoch(train_datasets)
            val_acc = self.val_epoch(val_datasets)
            print("epoch_id: {}, train acc is: {}, val acc is {}".format(i, train_acc, val_acc))
            self.save() # 每个epoch保存一次模型


# 训练配置
epochs = 10
lr = 0.01
model_path = './mnist.pdparams'

train_loader = load_data(mode='train')

val_loader = load_data(mode='eval')

model = resnet50
opt = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())

trainer = Trainer(
    model_path=model_path,
    model=model,
    optimizer=opt
)


# 训练过程
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

trainer.train(train_datasets=train_loader, val_datasets=val_loader, epochs=epochs)
