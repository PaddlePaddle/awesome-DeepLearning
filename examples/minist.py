#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pwd')


# In[2]:


import numpy as np
import paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os


# In[3]:


BUF_SIZE = 512
BATCH_SIZE = 128

# 读取官方训练集
trainData = paddle.dataset.mnist.train()
# 读取官方测试集
testData = paddle.dataset.mnist.test()


trainReader = paddle.batch(
    paddle.reader.shuffle(
        trainData,
        buf_size = BUF_SIZE
    ), 
    batch_size = BATCH_SIZE
)


testReader = paddle.batch(
    paddle.reader.shuffle(
        testData,
        buf_size = BUF_SIZE
    ),
    batch_size = BATCH_SIZE
)

# sampleData = next(trainData())

# print(sampleData)


# ## Step2 配置网络结构&定义损失和优化函数

# In[4]:


# 网络结构1
# 使用并定义多层感知器
def multilayerPerception(img):
    # 第一个全连接层，激活函数为ReLU
    layer1 = fluid.layers.fc(input=img, size=1000, act='relu')
    # 第二个全连接层，激活函数为ReLU
    layer2 = fluid.layers.fc(input=layer1, size=1000, act='relu')
    # 全连接输出层，激活函数为softmax，输出大小为10
    prediction = fluid.layers.fc(input=layer2, size=10, act='softmax')

    return prediction


# In[5]:


# 网络结构2
# 使用并定义卷积层
def cnn(img):
    convPool1 = fluid.nets.simple_img_conv_pool(
        # 输入图像
        input=img,
        # 卷积核大小
        filter_size=5,
        # 卷积核数目
        num_filters=20,
        # 池化层大小
        pool_size=2,
        # 池化层步长
        pool_stride=2,
        # 激活类型
        act='relu'
    )
    convPool2 = fluid.nets.simple_img_conv_pool(
        # 以第一个卷积层为输入
        input=convPool1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )
    convPool3 = fluid.nets.simple_img_conv_pool(
        # 以第二个卷积层为输入
        input=convPool2,
        filter_size=4,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )

    # 以softmax为激活函数的全连接输出层
    # 输出10个数字
    prediction = fluid.layers.fc(input=convPool3, size=10, act='softmax')

    return prediction


# In[6]:


# 网络结构3
# 尝试使用VGG模型
def conv_block(inp, numFilter, groups, dropouts):
    return fluid.nets.img_conv_group(
        input=inp,
        pool_size=2,
        pool_stride=2,
        conv_num_filter=[numFilter] * groups,
        conv_filter_size=3,
        conv_act='relu',
        conv_with_batchnorm=True,
        conv_batchnorm_drop_rate=dropouts,
        pool_type='max'
    )

def vgg(input):
    conv1 = conv_block(input, 64, 2, [0,0])
    conv2 = conv_block(conv1, 128, 2, [0,0])
    conv3 = conv_block(conv2, 128, 3, [0,0,0])
    drop = fluid.layers.dropout(x=conv3, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.0)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)

    # 全连接输出层，以softmax为激活函数，大小为10
    prediction = fluid.layers.fc(input=fc2, size=10, act='softmax')

    return prediction


# In[7]:


# 定义输入输出层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


# In[8]:


# 调用定义好的网络获取分类器
# 网络结构1
#model = multilayerPerception(image)

# 网络结构2
model = cnn(image)

# 网络结构3
#model = vgg(image)


# In[9]:


# 获取损失函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avgCost = fluid.layers.mean(cost)
# 获取准确率函数
acc = fluid.layers.accuracy(input=model, label=label)


# In[10]:


# 定义优化方法
learnRate = 0.001

# 采用Adam算法进行优化
myOptimizer = fluid.optimizer.AdamOptimizer(learning_rate=learnRate)

# 采用Adamax算法进行优化
#myOptimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=learnRate)

# 采用Adagrad算法进行优化
#myOptimizer = fluid.optimizer.AdagradOptimizer(learning_rate=learnRate)

# 采用DecayedAdagrad算法进行优化
#myOptimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=learnRate)

# 采用Lamb算法进行优化
#myOptimizer = fluid.optimizer.LambOptimizer(learning_rate=learnRate)

# 采用SGD算法进行优化
#myOptimizer = fluid.optimizer.SGDOptimizer(learning_rate=learnRate)

opts = myOptimizer.minimize(avgCost)


# ## Step3：模型训练

# In[11]:


# 定义一个使用CPU的解析器
# place = fluid.CPUPlace()
# 定义一个使用GPU的解析器
place = fluid.CUDAPlace(0)


exe = fluid.Executor(place)
# 参数初始化
exe.run(fluid.default_startup_program())


# In[12]:


# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])


# In[13]:


# 为作图做数据准备
allTrainIter = 0
allTrainIters = []
allTrainCosts = []
allTrainAccs = []


# In[14]:


# 模型存储地址
modelSavePath = 'home/aistudio/data/model'


# In[15]:


# 开始训练和测试
trainTimes = 5
for timeId in range(trainTimes):
    # 训练
    for batchId, data in enumerate(trainReader()):
        trainCost, trainAcc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avgCost, acc]
        )

        allTrainIter += 128
        allTrainIters.append(allTrainIter)
        allTrainCosts.append(trainCost[0])
        allTrainAccs.append(trainAcc[0])

        # 每100次batch打印一次信息
        if batchId % 100 == 0:
            print("TimeId:%d, Batch:%d, Cost:%0.5f, Acc:%0.5f" %(timeId, batchId, trainCost[0], trainAcc[0]))
    # 测试
    testAccs = []
    testCosts = []

    #每训练一轮，就测试一轮
    for batchId, data in enumerate(testReader()):
        testCost, testAcc = exe.run(
            program=fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avgCost, acc]
        )
        testAccs.append(testAcc[0])
        testCosts.append(testCost[0])

    # 求测试结果的平均值
    testCost = sum(testCosts) / len(testCosts)
    testAcc = sum(testAccs) / len(testAccs)
    print("TimeId(Test):%d, Cost:%0.5f, Acc:%0.5f" %(timeId, testCost, testAcc))

    
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)
    print('Save Model to %s' %(modelSavePath))

    fluid.io.save_inference_model(
        modelSavePath,
        ['image'],
        [model],
        exe
    )

# 画图
plt.title('Training', fontsize=24)
plt.xlabel('iter', fontsize=20)
plt.ylabel('cost/acc', fontsize=20)
plt.plot(allTrainIters, allTrainCosts, color='red', label='Train Cost')
plt.plot(allTrainIters, allTrainAccs, color='blue', label='Train Acc')
plt.legend()
plt.grid()
plt.show()
        


# ## Step4 模型预测

# In[16]:


# !cat /home/aistudio/data/data5562/4.png
# 查看需要预测的图片
predictImgPath = '/home/aistudio/data/data37667/4.png'
img = Image.open(predictImgPath).convert('L').resize((28, 28),Image.ANTIALIAS)
plt.imshow(img)
plt.show()


# In[17]:


# 图片预处理
def ImagePreprocess(path):
    img = Image.open(path).convert('L').resize((28, 28), Image.ANTIALIAS)
    img = np.array(img).reshape(1, 1, 28, 28).astype(np.float32)
    img = img / 255.0 * 2.0 - 1.0
    # print(img)

    return img

# ImagePreprocess('/home/aistudio/data/data5562/4.png')


# In[18]:


inferExe = fluid.Executor(place)
inferenceScope = fluid.core.Scope()

# 加载图片数据并开始预测
with fluid.scope_guard(inferenceScope):
    # 获取训练好的模型
    [
        inferenceProgram,
        feedTragetNames,
        fetchTargets
    ] = fluid.io.load_inference_model(modelSavePath, inferExe)

    img = ImagePreprocess(predictImgPath)

    results = exe.run(
        program=inferenceProgram,
        feed={feedTragetNames[0]: img},
        fetch_list=fetchTargets
    )


lab = np.argsort(results)

img = Image.open(predictImgPath)
plt.imshow(img)
plt.show()
print('该图片的预测结果的label为%d' %(lab[0][0][-1]))

