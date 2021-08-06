import paddle
import paddle.vision.transforms as T

transforms = T.Compose([#数据增强
                T.RandomHorizontalFlip(0.5),#随机水平翻转
                T.RandomVerticalFlip(0.15),#随机垂直翻转
                T.RandomRotation(15),#随机旋转角度
                T.ToTensor()#数据的格式转换和标准化 HWC => CHW
])

train_dataset = paddle.vision.datasets.Cifar100(mode='train', transform=transforms)#训练数据集
test_dataset = paddle.vision.datasets.Cifar100(mode='test', transform=transforms)#测试数据集

vgg16 = paddle.vision.models.vgg16(pretrained=True)#模型开发，调用高层API，使用vgg16网络

model = paddle.Model(vgg16)

# 模型训练相关配置，设置学习率，损失计算方法，优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(learning_rate=0.1,parameters=model.parameters()),
                                    paddle.nn.CrossEntropyLoss(),
                                    paddle.metric.Accuracy())

#开始模型训练，用训练数据集进行训练，设置训练总轮数，batch包含的样本数等参数
model.fit(train_dataset,batch_size=512,epochs=30,verbose=1,save_dir="log/vgg/",shuffle=True)
#记录结果
result = model.evaluate(test_dataset, verbose=1,batch_size=512)
print(result)
model.save('/model/vgg') 
