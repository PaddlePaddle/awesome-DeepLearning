import paddle
paddle.__version__


import paddle.vision.transforms as T

# 训练数据集
train_dataset = paddle.vision.datasets.Cifar100(mode='train', transform=T.ToTensor())

# 验证数据集
eval_dataset = paddle.vision.datasets.Cifar100(mode='test', transform=T.ToTensor())
#数据增强部分
train_transforms = T.Compose([T.RandomHorizontalFlip(0.5),#水平翻转
                        T.RandomRotation(15),#随机反转角度范围
                        T.RandomVerticalFlip(0.15),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],data_format='CHW', to_rgb=True)])

eval_transforms = T.Compose([T.RandomHorizontalFlip(0.5),#水平翻转
                        T.RandomRotation(15),#随机反转角度范围
                        T.RandomVerticalFlip(0.15),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],data_format='CHW', to_rgb=True)])

print('训练集样本量: {}，验证集样本量: {}'.format(len(train_dataset), len(eval_dataset)))


#选用ResNet50网络，CIFAR100最后的分类数为100
network = paddle.vision.models.resnet50(num_classes=100)

#模型封装
model = paddle.Model(network)
#模型可视化
model.summary((-1, 3, 32, 32))
#优化器配置
model.prepare(paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters()),#使用Adam优化器，学习率为0.0001
              paddle.nn.CrossEntropyLoss(),#损失函数使用交叉熵函数
              paddle.metric.Accuracy()) #Acc用top1与top5精准度表示


#开始模型训练
model.fit(train_dataset,
          eval_dataset,
          epochs=20,#训练的轮数
          batch_size=256,#每次训练多少个
          verbose=1,#显示模式
          shuffle=True,#打乱数据集顺序
          )
result = model.evaluate(eval_dataset, verbose=1,batch_size=256)

print(result)


model.save('Restnet_Test')