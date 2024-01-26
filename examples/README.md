# 项目地址：https://aistudio.baidu.com/aistudio/projectdetail/2247798
# DenseNet结构及原理概述
## 1.网络特点
深度学习网络中，随着网络深度的加深，梯度消失的问题会越来越明显。ResNet，Highway Networks，Stochastic depth，FractalNets等网络都在不同方面针对这个问题提出解决方案，但核心方法都是**建立浅层与深层之间的连接**。

![](https://ai-studio-static-online.cdn.bcebos.com/0511a5d8389644e3bc31cccab413e01bada1b1abf13e41719e8e2e7830d51a10)

DenseNet继续延申了这一思想，将当前层与之前所有层连接起来，上图即为一个Dense Block。
DenseNet的一个优点就是网络更窄，参数更少，并且特征和梯度的传递更有效，网络也就更容易训练。
## 2.传递公式
先看ResNet的：

![](https://ai-studio-static-online.cdn.bcebos.com/b324ab87677740af8939bcac93f62104d727dbd5d84243c1ae58fc5189bff1f1)

将上一层的输入和输出相加得到这一层的输入。
而DenseNet:

![](https://ai-studio-static-online.cdn.bcebos.com/343f7c85401545698e88d241d4583093fa30c3bde32f4f149f532f70ea934c15)

这一层的输入是之前的所有层，从式子中就能清晰地看出DenseNet的运作方式以及和ResNet的差别。
## 3.网络整体结构图

![](https://ai-studio-static-online.cdn.bcebos.com/6537a31b773943f9af75c534a9a779ccadef505cb29f4a67a9937894946059a7)

图中包含3个Dense Block,可以看到每个Dense Block中的所有层都与其之前的每一层相连。每两个Dense Block之间还有一个$1 \times 1$卷积层和一个$2 \times 2$池化层,这是为了减少输入的feature map，降维减少计算量，融合各通道特征。
每张图片先经过卷积输入，然后经过几个Dense Block，最后再经过一次卷积输入到全连接层中，输出（分类）结果。
## 4.几种常用的结构

![](https://ai-studio-static-online.cdn.bcebos.com/2918cb1651f14367a68f52aa165b6e7101bdb91c521a42678cfba367f1c24620)

图中的k=32和k=48表示每个Dense Block中每层输出的feature map个数，作者的实验表明32或48这种较小的k会有更好的效果。
## 5.网络效果

![](https://ai-studio-static-online.cdn.bcebos.com/0c287b101fc543f6a89599b7ce32e0d40637f573e07c46bfa45b84a66139942c)

Table2是在三个数据集（C10，C100，SVHN）上和其他算法的对比结果。ResNet[11]就是kaiming He的论文，对比结果一目了然。DenseNet-BC的网络参数和相同深度的DenseNet相比确实减少了很多！参数减少除了可以节省内存，还能减少过拟合。这里对于SVHN数据集，DenseNet-BC的结果并没有DenseNet(k=24)的效果好，作者认为原因主要是SVHN这个数据集相对简单，更深的模型容易过拟合。在表格的倒数第二个区域的三个不同深度L和k的DenseNet的对比可以看出随着L和k的增加，模型的效果是更好的。

![](https://ai-studio-static-online.cdn.bcebos.com/7c4c1f964aca4b0795e4b9e7eaa00c3f2f41c12eb7c04f8181a67000eaee2473)

左图是参数复杂度和错误率的对比，可以在相同错误率下看参数复杂度，也可以在相同参数复杂度下看错误率，提升还是很明显的。右边是flops（可以理解为计算复杂度）和错误率的对比，同样有效果。

参考文献：[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

# 代码实现
## 1. 实验设计逻辑
### 解释任务，说明实验设计逻辑
任务要求是在眼疾识别数据集上训练DenseNet网络，实现分类的效果。
根据要求构建Densenet网络并使用眼疾识别数据集中的训练集进行训练，然后再用测试集测试训练效果。

## 2. 数据处理
### 解释数据集，处理数据为模型输入格式
iChallenge-PM是百度大脑和中山大学中山眼科中心联合举办的iChallenge比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。  
training.zip：包含训练中的图片和标签。  
validation.zip：包含验证集的图片。  
valid_gt.zip：包含验证集的标签。  
iChallenge-PM中既有病理性近视患者的眼底图片，也有非病理性近视患者的图片，命名规则如下：  
病理性近视（PM）：文件名以P开头。  
非病理性近视（non-PM）： 高度近似（high myopia）：文件名以H开头。  
正常眼睛（normal）：文件名以N开头。  
处理数据集：根据数据集介绍，训练集images通过文件名获取相应的labels,测试集通过读取PM_Label_and_Fovea_Location.xlsx文件获取文件名和label信息，然后分别生成traindata.txt和valdata.txt文件。使用时直接读入文件即可获取图片与标签对应关系。


```python
#生成label文本
import os
import numpy as np
path = ''
trainpath = 'PALM-Training400'
imgdirs = os.listdir(os.path.join(path, trainpath))
traindata = open('traindata.txt', 'w', encoding = 'utf-8')
for file in imgdirs:
    if '.jpg' in file:
        traindata.write(trainpath + '/' + file + ' ')
        if file[0] is 'H' or file[0] is 'N':
            traindata.write('0')
        else:
            traindata.write('1')
        traindata.write('\n')
traindata.flush()
traindata.close()
```


```python
#生成验证集label文本，通过读取xlsx文件
import os
import numpy as np
from openpyxl import load_workbook
path = ''
valpath = 'PALM-Validation400'
vallabelpath = 'PM_Label_and_Fovea_Location.xlsx'
oldfile = load_workbook(os.path.join(path, vallabelpath))
newfile = open('valdata.txt', 'w', encoding = 'utf-8')
sheet = oldfile.worksheets[0]
rows = sheet.rows
for row in sheet[2:401]:
    newfile.write(valpath + '/' + row[1].value + ' ' + str(row[2].value) + '\n')
newfile.flush()
newfile.close()
```


```python

```


```python
%cd work/datasets/
```

    /home/aistudio/work/datasets


## 3. 训练配置
### 定义模型训练的超参数，模型实例化，指定训练的 cpu 或 gpu 资 源，定义优化器等等


```python
# -*- coding: UTF-8 -*-
"""
训练常用视觉基础网络，用于分类任务
需要将训练图片，类别文件 label_list.txt 放置在同一个文件夹下
程序会先读取 train.txt 文件获取类别数和图片数量
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.75'
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid
import codecs
import logging

from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from PIL import Image
from PIL import ImageEnhance

train_parameters = {  
    "input_size": [3, 512, 512],
    "class_dim": 2,  # 分类数，会在初始化自定义 reader 的时候获得  
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得  
    "label_dict": {},
    "data_dir": "",  # 训练数据存储地址  
    "train_file_list": "traindata.txt",
    "label_file": "label_list.txt",
    "save_freeze_dir": "./freeze-model4",
    "save_persistable_dir": "./persistable-params",
    "continue_train": True,        # 是否接着上一次保存的参数接着训练，优先级高于预训练模型  
    "pretrained": False,            # 是否使用预训练的模型  
    "pretrained_dir": None,
    # "data/data6593/DenseNet_pretrained",
    "mode": "train",
    "num_epochs": 100,
    "train_batch_size": 20,
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值  
    "use_gpu": True,
    "dropout_prob": 0.2,
    "dropout_seed": None,
    "image_enhance_strategy": {  # 图像增强相关策略  
        "need_distort": True,  # 是否启用图像颜色增强  
        "need_rotate": True,   # 是否需要增加随机角度  
        "need_crop": True,      # 是否要增加裁剪  
        "need_flip": True,      # 是否要增加水平随机翻转  
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125,
        "rotate_prob": 0.5,
        "rotate_range": 14
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 5,
        "good_acc1": 0.96
    },
    "rsm_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    }
}
```

## 4. 模型设计
### 根据任务设计模型，需要给出模型设计图
这里采用DenseNet-121结构（结构数据前文图片中有)  
[网络结构图](./work/datasets/__model__.svg)


```python
class DenseNet():
    def __init__(self, layers, dropout_prob):
        self.layers = layers
        self.dropout_prob = dropout_prob

    def bottleneck_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num * 4, filter_size=1, name=name + '_conv1')
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        bn = fluid.layers.batch_norm(input=dropout, act='relu', name=name + '_bn2')
        conv2 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=3, padding=1, name=name + '_conv2')
        dropout = fluid.layers.dropout(x=conv2, dropout_prob=self.dropout_prob)

        return dropout

    def dense_block(self, input, block_num, fliter_num, name):
        layers = []
        layers.append(input)#拼接到列表

        x = self.bottleneck_layer(input, fliter_num, name=name + '_bottle_' + str(0))
        layers.append(x)
        for i in range(block_num - 1):
            x = paddle.fluid.layers.concat(layers, axis=1)
            x = self.bottleneck_layer(x, fliter_num, name=name + '_bottle_' + str(i + 1))
            layers.append(x)

        return paddle.fluid.layers.concat(layers, axis=1)

    def transition_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=1, name=name + '_conv1')
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        return fluid.layers.pool2d(input=dropout, pool_size=2, pool_type='avg', pool_stride=2)

    def net(self, input, class_dim=2):

        layer_count_dict = {
            121: (32, [6, 12, 24, 16]),
            169: (32, [6, 12, 32, 32]),
            201: (32, [6, 12, 48, 32]),
            161: (48, [6, 12, 36, 24])
        }
        layer_conf = layer_count_dict[self.layers]

        conv = fluid.layers.conv2d(input=input, num_filters=layer_conf[0] * 2,
            filter_size=7, stride=2, padding=3, name='densenet_conv0')
        conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_padding=1, pool_type='max', pool_stride=2)
        for i in range(len(layer_conf[1]) - 1):
            conv = self.dense_block(conv, layer_conf[1][i], layer_conf[0], 'dense_' + str(i))
            conv = self.transition_layer(conv, layer_conf[0], name='trans_' + str(i))

        conv = self.dense_block(conv, layer_conf[1][-1], layer_conf[0], 'dense_' + str(len(layer_conf[1])))
        conv = fluid.layers.pool2d(input=conv, global_pooling=True, pool_type='avg')
        out = fluid.layers.fc(conv, class_dim, act='softmax')
        # last fc layer is "out"
        return out



```


```python
def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)



```


```python
def resize_img(img, target_size):
    """
    强制缩放图片
    :param img:
    :param target_size:
    :return:
    """
    target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def random_crop(img, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((train_parameters['input_size'][1], train_parameters['input_size'][2]), Image.BILINEAR)
    return img


def rotate_image(img):
    """
    图像增强，增加随机旋转角度
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['rotate_prob']:
        range = train_parameters['image_enhance_strategy']['rotate_range']
        angle = np.random.randint(-range, range)
        img = img.rotate(angle)
    return img


def random_brightness(img):
    """
    图像增强，亮度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    图像增强，对比度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    图像增强，饱和度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    图像增强，色度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img):
    """
    概率的图像增强
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.35:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    elif prob < 0.7:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img



```


```python
def custom_image_reader(file_list, data_dir, mode):
    """
    自定义用户图片读取器，先初始化图片种类，数量
    :param file_list:
    :param data_dir:
    :param mode:
    :return:
    """
    with codecs.open(file_list) as flist:
        lines = [line.strip() for line in flist]

    def reader():
        np.random.shuffle(lines)
        for line in lines:
            if mode == 'train' or mode == 'val':
                img_path, label = line.split()
                img = Image.open(img_path)
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if train_parameters['image_enhance_strategy']['need_distort'] == True:
                        img = distort_color(img)
                    if train_parameters['image_enhance_strategy']['need_rotate'] == True:
                        img = rotate_image(img)
                    if train_parameters['image_enhance_strategy']['need_crop'] == True:
                        img = random_crop(img, train_parameters['input_size'])
                    if train_parameters['image_enhance_strategy']['need_flip'] == True:
                        mirror = int(np.random.uniform(0, 2))
                        if mirror == 1:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    # HWC--->CHW && normalized
                    img = np.array(img).astype('float32')
                    img -= train_parameters['mean_rgb']
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img *= 0.007843                 # 像素值归一化
                    yield img, int(label)
                except Exception as e:
                    pass                            # 以防某些图片读取处理出错，加异常处理
            elif mode == 'test':
                img_path = os.path.join(data_dir, line)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = resize_img(img, train_parameters['input_size'])
                # HWC--->CHW && normalized
                img = np.array(img).astype('float32')
                img -= train_parameters['mean_rgb']
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img *= 0.007843  # 像素值归一化
                yield img

    return reader



```


```python
def optimizer_rms_setting():
    """
    阶梯型的学习率适合比较大规模的训练数据
    """
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values))

    return optimizer



```


```python
def load_params(exe, program):
    if train_parameters['continue_train'] and os.path.exists(train_parameters['save_persistable_dir']):
        logger.info('load params from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_persistable_dir'],
                                   main_program=program)
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_dir']):
        logger.info('load params from pretrained model')
        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_dir'], main_program=program,
                           predicate=if_exist)



```

## 5. 模型训练与评估
### 训练模型，在训练过程中，根据开发集适时打印结果
模型文件保存在./freeze-model中(此次训练最终的模型在./freeze-model4中)  
日志文件保存在./logs/train.log中(此次训练最终的训练日志与评估结果在./logs/train4.log中)


```python
def train():
    train_prog = fluid.Program()
    train_startup = fluid.Program()
    logger.info("create prog success")
    logger.info("train config: %s", str(train_parameters))
    logger.info("build input custom reader and data feeder")
    file_list = os.path.join(train_parameters['data_dir'], "traindata.txt")
    mode = train_parameters['mode']
    batch_reader = paddle.batch(custom_image_reader(file_list, train_parameters['data_dir'], mode),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=False)
    batch_reader = paddle.reader.shuffle(batch_reader, train_parameters['train_batch_size'])
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
    # 定义输入数据的占位符
    paddle.enable_static()
    img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    print(feeder)

    # 选取不同的网络
    logger.info("build newwork")
    model = DenseNet(121, train_parameters['dropout_prob'])
    out = model.net(input=img, class_dim=train_parameters['class_dim'])

    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    optimizer = optimizer_rms_setting()
    optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    main_program = fluid.default_main_program()
    exe.run(fluid.default_startup_program())
    train_fetch_list = [avg_cost.name, acc_top1.name, out.name]

    load_params(exe, main_program)

    # 训练循环主体
    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    good_acc1 = stop_strategy['good_acc1']
    successive_count = 0
    stop_train = False
    total_batch_count = 0
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        batch_id = 0
        for step_id, data in enumerate(batch_reader()):
            t1 = time.time()
            loss, acc1, pred_ot = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=train_fetch_list)
            t2 = time.time()
            batch_id += 1
            total_batch_count += 1
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            if batch_id % 10 == 0:
                logger.info("Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, time {4}".format(pass_id, batch_id, loss, acc1,
                                                                                            "%2.2f sec" % period))
            # 简单的提前停止策略，认为连续达到某个准确率就可以停止了
            if acc1 >= good_acc1:
                successive_count += 1
                logger.info("current acc1 {0} meets good {1}, successive count {2}".format(acc1, good_acc1, successive_count))
                fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program,
                                              executor=exe)
                if successive_count >= successive_limit:
                    logger.info("end training")
                    stop_train = True
                    break
            else:
                successive_count = 0

            # 通用的保存策略，减小意外停止的损失
            if total_batch_count % sample_freq == 0:
                logger.info("temp save {0} batch train result, current acc1 {1}".format(total_batch_count, acc1))
                fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
        if stop_train:
            break
    logger.info("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
    fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program,
                                              executor=exe)


```


```python
if __name__ == '__main__':
    init_log_config()
    init_train_parameters()
    train()
```

    2021-08-03 18:02:49,533 - <ipython-input-18-f06b761b0d12>[line:4] - INFO: create prog success
    2021-08-03 18:02:49,534 - <ipython-input-18-f06b761b0d12>[line:5] - INFO: train config: {'input_size': [3, 512, 512], 'class_dim': 2, 'image_count': 400, 'label_dict': {'high': 0, 'PM': 1}, 'data_dir': '', 'train_file_list': 'traindata.txt', 'label_file': 'label_list.txt', 'save_freeze_dir': './freeze-model', 'save_persistable_dir': './persistable-params', 'continue_train': True, 'pretrained': False, 'pretrained_dir': None, 'mode': 'train', 'num_epochs': 100, 'train_batch_size': 20, 'mean_rgb': [127.5, 127.5, 127.5], 'use_gpu': True, 'dropout_prob': 0.2, 'dropout_seed': None, 'image_enhance_strategy': {'need_distort': True, 'need_rotate': True, 'need_crop': True, 'need_flip': True, 'hue_prob': 0.5, 'hue_delta': 18, 'contrast_prob': 0.5, 'contrast_delta': 0.5, 'saturation_prob': 0.5, 'saturation_delta': 0.5, 'brightness_prob': 0.5, 'brightness_delta': 0.125, 'rotate_prob': 0.5, 'rotate_range': 14}, 'early_stop': {'sample_frequency': 50, 'successive_limit': 5, 'good_acc1': 0.96}, 'rsm_strategy': {'learning_rate': 0.001, 'lr_epochs': [20, 40, 60, 80, 100], 'lr_decay': [1, 0.5, 0.25, 0.1, 0.05, 0.01]}}
    2021-08-03 18:02:49,535 - <ipython-input-18-f06b761b0d12>[line:6] - INFO: build input custom reader and data feeder
    2021-08-03 18:02:49,536 - <ipython-input-18-f06b761b0d12>[line:22] - INFO: build newwork


    <paddle.fluid.data_feeder.DataFeeder object at 0x7f8004a43810>


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:689: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      elif dtype == np.bool:
    2021-08-03 18:02:53,845 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 0, start read image
    2021-08-03 18:04:53,008 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 0, trainbatch 10, loss 0.7001757025718689, acc1 0.699999988079071, time 0.65 sec
    2021-08-03 18:04:59,785 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 0, trainbatch 20, loss 0.5529266595840454, acc1 0.699999988079071, time 0.64 sec
    2021-08-03 18:04:59,804 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 1, start read image
    2021-08-03 18:06:56,209 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 1, trainbatch 10, loss 1.736151933670044, acc1 0.6499999761581421, time 0.65 sec
    2021-08-03 18:07:03,022 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 1, trainbatch 20, loss 0.6324112415313721, acc1 0.75, time 0.65 sec
    2021-08-03 18:07:03,053 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 2, start read image
    2021-08-03 18:08:58,076 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 2, trainbatch 10, loss 0.5782172679901123, acc1 0.6499999761581421, time 0.64 sec
    2021-08-03 18:08:58,096 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 50 batch train result, current acc1 0.6499999761581421
    2021-08-03 18:09:06,545 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 2, trainbatch 20, loss 0.48193299770355225, acc1 0.8999999761581421, time 0.65 sec
    2021-08-03 18:09:06,567 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 3, start read image
    2021-08-03 18:10:59,534 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 3, trainbatch 10, loss 0.3248681128025055, acc1 0.8500000238418579, time 0.60 sec
    2021-08-03 18:11:06,576 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 3, trainbatch 20, loss 0.4310462772846222, acc1 0.800000011920929, time 0.98 sec
    2021-08-03 18:11:06,605 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 4, start read image
    2021-08-03 18:12:58,913 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 4, trainbatch 10, loss 1.2876142263412476, acc1 0.550000011920929, time 0.64 sec
    2021-08-03 18:13:04,464 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:13:07,725 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 4, trainbatch 20, loss 0.36517342925071716, acc1 0.8500000238418579, time 0.64 sec
    2021-08-03 18:13:07,743 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 100 batch train result, current acc1 0.8500000238418579
    2021-08-03 18:13:09,454 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 5, start read image
    2021-08-03 18:15:07,713 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 5, trainbatch 10, loss 0.3396274745464325, acc1 0.8500000238418579, time 0.66 sec
    2021-08-03 18:15:14,516 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 5, trainbatch 20, loss 0.5121028423309326, acc1 0.75, time 0.65 sec
    2021-08-03 18:15:14,538 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 6, start read image
    2021-08-03 18:17:06,319 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:17:09,529 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 6, trainbatch 10, loss 0.40974169969558716, acc1 0.800000011920929, time 0.59 sec
    2021-08-03 18:17:16,398 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 6, trainbatch 20, loss 0.5014650225639343, acc1 0.75, time 0.60 sec
    2021-08-03 18:17:16,419 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 7, start read image
    2021-08-03 18:19:03,092 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:19:05,356 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 18:19:09,239 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 7, trainbatch 10, loss 0.550878643989563, acc1 0.800000011920929, time 0.59 sec
    2021-08-03 18:19:09,259 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 150 batch train result, current acc1 0.800000011920929
    2021-08-03 18:19:17,025 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:19:19,841 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 7, trainbatch 20, loss 1.2134144306182861, acc1 0.44999998807907104, time 0.59 sec
    2021-08-03 18:19:19,859 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 8, start read image
    2021-08-03 18:21:10,494 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 8, trainbatch 10, loss 0.3217931389808655, acc1 0.8999999761581421, time 0.59 sec
    2021-08-03 18:21:11,685 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:21:19,070 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 8, trainbatch 20, loss 0.3484116494655609, acc1 0.8500000238418579, time 0.59 sec
    2021-08-03 18:21:19,089 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 9, start read image
    2021-08-03 18:23:19,555 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 9, trainbatch 10, loss 0.48367443680763245, acc1 0.75, time 0.59 sec
    2021-08-03 18:23:26,256 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 9, trainbatch 20, loss 0.30781298875808716, acc1 0.8999999761581421, time 0.60 sec
    2021-08-03 18:23:26,277 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 200 batch train result, current acc1 0.8999999761581421
    2021-08-03 18:23:27,869 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 10, start read image
    2021-08-03 18:25:17,392 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 10, trainbatch 10, loss 0.33601438999176025, acc1 0.75, time 0.98 sec
    2021-08-03 18:25:19,752 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:25:25,911 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 10, trainbatch 20, loss 0.4109388589859009, acc1 0.75, time 1.22 sec
    2021-08-03 18:25:25,913 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 11, start read image
    2021-08-03 18:27:13,212 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:27:20,203 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 11, trainbatch 10, loss 0.3686891794204712, acc1 0.75, time 0.99 sec
    2021-08-03 18:27:23,798 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:27:28,771 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 11, trainbatch 20, loss 0.49685603380203247, acc1 0.800000011920929, time 1.10 sec
    2021-08-03 18:27:28,774 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 12, start read image
    2021-08-03 18:29:12,852 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:29:19,789 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 12, trainbatch 10, loss 0.23170387744903564, acc1 0.949999988079071, time 0.99 sec
    2021-08-03 18:29:19,791 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 250 batch train result, current acc1 0.949999988079071
    2021-08-03 18:29:27,806 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 12, trainbatch 20, loss 0.2838474214076996, acc1 0.8500000238418579, time 0.60 sec
    2021-08-03 18:29:27,827 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 13, start read image
    2021-08-03 18:31:19,037 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:31:24,392 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 13, trainbatch 10, loss 0.4358825385570526, acc1 0.8999999761581421, time 0.65 sec
    2021-08-03 18:31:31,209 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 13, trainbatch 20, loss 0.4661249816417694, acc1 0.75, time 0.59 sec
    2021-08-03 18:31:31,229 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 14, start read image
    2021-08-03 18:33:26,718 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 14, trainbatch 10, loss 0.34411731362342834, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:33:33,399 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 14, trainbatch 20, loss 0.5190801620483398, acc1 0.75, time 0.60 sec
    2021-08-03 18:33:33,420 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 300 batch train result, current acc1 0.75
    2021-08-03 18:33:35,003 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 15, start read image
    2021-08-03 18:35:30,947 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 15, trainbatch 10, loss 0.2045409381389618, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:35:37,668 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 15, trainbatch 20, loss 0.5817281007766724, acc1 0.75, time 0.60 sec
    2021-08-03 18:35:37,691 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 16, start read image
    2021-08-03 18:37:26,391 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:37:33,979 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 16, trainbatch 10, loss 0.5690133571624756, acc1 0.75, time 0.66 sec
    2021-08-03 18:37:40,474 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 16, trainbatch 20, loss 0.20127500593662262, acc1 0.8999999761581421, time 0.59 sec
    2021-08-03 18:37:40,494 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 17, start read image
    2021-08-03 18:39:31,275 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 17, trainbatch 10, loss 0.7717211842536926, acc1 0.550000011920929, time 0.60 sec
    2021-08-03 18:39:31,296 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 350 batch train result, current acc1 0.550000011920929
    2021-08-03 18:39:36,823 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:39:41,190 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 17, trainbatch 20, loss 0.2284773290157318, acc1 0.8999999761581421, time 0.60 sec
    2021-08-03 18:39:41,210 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 18, start read image
    2021-08-03 18:41:29,688 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 18, trainbatch 10, loss 0.37405675649642944, acc1 0.8500000238418579, time 0.60 sec
    2021-08-03 18:41:36,076 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 18, trainbatch 20, loss 0.37402376532554626, acc1 0.75, time 0.60 sec
    2021-08-03 18:41:36,095 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 19, start read image
    2021-08-03 18:43:19,146 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:43:26,485 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 19, trainbatch 10, loss 0.3060307502746582, acc1 0.8500000238418579, time 0.61 sec
    2021-08-03 18:43:28,675 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:43:35,016 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 19, trainbatch 20, loss 0.3649413287639618, acc1 0.8500000238418579, time 0.59 sec
    2021-08-03 18:43:35,037 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 400 batch train result, current acc1 0.8500000238418579
    2021-08-03 18:43:36,609 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 20, start read image
    2021-08-03 18:45:28,089 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 20, trainbatch 10, loss 0.281959593296051, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:45:29,906 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:45:36,706 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 20, trainbatch 20, loss 0.3690020442008972, acc1 0.75, time 0.60 sec
    2021-08-03 18:45:36,729 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 21, start read image
    2021-08-03 18:47:23,374 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:47:30,322 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 21, trainbatch 10, loss 0.23612964153289795, acc1 0.949999988079071, time 0.65 sec
    2021-08-03 18:47:36,926 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 21, trainbatch 20, loss 0.2393002063035965, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:47:36,948 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 22, start read image
    2021-08-03 18:49:27,660 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:49:33,713 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 22, trainbatch 10, loss 0.2962583303451538, acc1 0.8500000238418579, time 0.65 sec
    2021-08-03 18:49:33,735 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 450 batch train result, current acc1 0.8500000238418579
    2021-08-03 18:49:42,036 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 22, trainbatch 20, loss 0.525825560092926, acc1 0.800000011920929, time 0.60 sec
    2021-08-03 18:49:42,057 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 23, start read image
    2021-08-03 18:51:35,935 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 23, trainbatch 10, loss 0.5031872391700745, acc1 0.75, time 0.99 sec
    2021-08-03 18:51:38,896 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:51:44,424 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 23, trainbatch 20, loss 0.17848190665245056, acc1 0.8999999761581421, time 1.16 sec
    2021-08-03 18:51:44,427 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 24, start read image
    2021-08-03 18:53:34,933 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 24, trainbatch 10, loss 0.14931993186473846, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:53:41,634 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 24, trainbatch 20, loss 0.14803865551948547, acc1 1.0, time 0.98 sec
    2021-08-03 18:53:41,637 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:53:43,304 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 500 batch train result, current acc1 1.0
    2021-08-03 18:53:44,984 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 25, start read image
    2021-08-03 18:55:35,691 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 25, trainbatch 10, loss 0.5347169041633606, acc1 0.75, time 0.66 sec
    2021-08-03 18:55:41,878 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:55:44,579 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 25, trainbatch 20, loss 0.41154026985168457, acc1 0.8500000238418579, time 0.65 sec
    2021-08-03 18:55:44,604 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 26, start read image
    2021-08-03 18:57:32,522 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:57:40,177 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 26, trainbatch 10, loss 0.2955395579338074, acc1 0.800000011920929, time 0.60 sec
    2021-08-03 18:57:46,482 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 26, trainbatch 20, loss 0.2001594603061676, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:57:46,502 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 27, start read image
    2021-08-03 18:59:32,631 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 18:59:35,491 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 27, trainbatch 10, loss 0.1807631552219391, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 18:59:35,512 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 550 batch train result, current acc1 0.949999988079071
    2021-08-03 18:59:44,382 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 27, trainbatch 20, loss 0.21316714584827423, acc1 0.8999999761581421, time 0.60 sec
    2021-08-03 18:59:44,404 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 28, start read image
    2021-08-03 19:01:27,468 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:01:32,377 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:01:36,978 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 28, trainbatch 10, loss 0.26947152614593506, acc1 0.949999988079071, time 0.59 sec
    2021-08-03 19:01:43,886 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 28, trainbatch 20, loss 0.3298850357532501, acc1 0.800000011920929, time 0.60 sec
    2021-08-03 19:01:43,907 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 29, start read image
    2021-08-03 19:03:24,247 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:03:28,205 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:03:30,431 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:03:33,166 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 29, trainbatch 10, loss 0.19889244437217712, acc1 0.949999988079071, time 0.58 sec
    2021-08-03 19:03:39,673 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 29, trainbatch 20, loss 0.32935017347335815, acc1 0.8500000238418579, time 0.59 sec
    2021-08-03 19:03:39,692 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 600 batch train result, current acc1 0.8500000238418579
    2021-08-03 19:03:41,646 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 30, start read image
    2021-08-03 19:05:31,243 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:05:38,623 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 30, trainbatch 10, loss 0.14522269368171692, acc1 0.949999988079071, time 0.61 sec
    2021-08-03 19:05:39,806 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:05:47,115 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 30, trainbatch 20, loss 0.272271990776062, acc1 0.949999988079071, time 0.58 sec
    2021-08-03 19:05:47,135 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 31, start read image
    2021-08-03 19:07:37,088 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 31, trainbatch 10, loss 0.1388687640428543, acc1 0.949999988079071, time 0.60 sec
    2021-08-03 19:07:38,269 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:07:42,022 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:07:47,716 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 31, trainbatch 20, loss 0.17286920547485352, acc1 0.8999999761581421, time 0.60 sec
    2021-08-03 19:07:47,737 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 32, start read image
    2021-08-03 19:09:35,110 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:09:41,785 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 32, trainbatch 10, loss 0.04632396250963211, acc1 1.0, time 0.58 sec
    2021-08-03 19:09:41,805 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:09:43,454 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 650 batch train result, current acc1 1.0
    2021-08-03 19:09:46,904 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:09:51,026 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:09:56,504 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 32, trainbatch 20, loss 0.17074042558670044, acc1 1.0, time 0.63 sec
    2021-08-03 19:09:56,522 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:09:58,192 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 33, start read image
    2021-08-03 19:11:56,385 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:12:04,031 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 33, trainbatch 10, loss 0.22037675976753235, acc1 0.8999999761581421, time 0.65 sec
    2021-08-03 19:12:08,333 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:12:12,878 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 33, trainbatch 20, loss 0.0755956768989563, acc1 1.0, time 0.65 sec
    2021-08-03 19:12:12,898 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:12:15,134 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 34, start read image
    2021-08-03 19:13:57,494 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:14:03,092 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:14:05,770 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:14:09,481 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:14:11,807 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 34, trainbatch 10, loss 0.9732007384300232, acc1 0.6499999761581421, time 0.63 sec
    2021-08-03 19:14:19,238 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 34, trainbatch 20, loss 0.41737303137779236, acc1 0.800000011920929, time 0.65 sec
    2021-08-03 19:14:19,257 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 700 batch train result, current acc1 0.800000011920929
    2021-08-03 19:14:20,962 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 35, start read image
    2021-08-03 19:16:10,002 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:16:13,187 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:16:19,561 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 35, trainbatch 10, loss 0.14571121335029602, acc1 0.8999999761581421, time 0.64 sec
    2021-08-03 19:16:20,551 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:16:24,644 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:16:30,528 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 35, trainbatch 20, loss 0.11610834300518036, acc1 1.0, time 0.60 sec
    2021-08-03 19:16:30,547 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:16:32,198 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 36, start read image
    2021-08-03 19:18:20,494 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:18:23,932 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:18:28,610 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 36, trainbatch 10, loss 0.2533027231693268, acc1 0.8500000238418579, time 0.61 sec
    2021-08-03 19:18:29,220 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:18:37,238 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 36, trainbatch 20, loss 0.13184118270874023, acc1 1.0, time 0.60 sec
    2021-08-03 19:18:37,259 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:18:38,903 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 37, start read image
    2021-08-03 19:20:26,606 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:20:29,528 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:20:33,607 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:20:36,422 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:20:40,865 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 37, trainbatch 10, loss 0.263095885515213, acc1 0.8999999761581421, time 0.69 sec
    2021-08-03 19:20:40,886 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 750 batch train result, current acc1 0.8999999761581421
    2021-08-03 19:20:46,926 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:20:51,657 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 37, trainbatch 20, loss 0.3562985360622406, acc1 0.800000011920929, time 0.62 sec
    2021-08-03 19:20:51,680 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 38, start read image
    2021-08-03 19:22:44,833 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:22:49,426 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 38, trainbatch 10, loss 0.045135386288166046, acc1 1.0, time 0.61 sec
    2021-08-03 19:22:49,451 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:22:54,736 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:22:57,334 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:23:01,016 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:23:04,067 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 38, trainbatch 20, loss 0.17996768653392792, acc1 0.949999988079071, time 0.66 sec
    2021-08-03 19:23:04,090 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 39, start read image
    2021-08-03 19:24:47,874 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:24:55,570 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 39, trainbatch 10, loss 0.07450184971094131, acc1 1.0, time 0.61 sec
    2021-08-03 19:24:55,589 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:24:59,429 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:25:02,835 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:25:05,634 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:25:09,350 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:25:12,386 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 39, trainbatch 20, loss 0.1687900871038437, acc1 0.949999988079071, time 0.65 sec
    2021-08-03 19:25:12,406 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 800 batch train result, current acc1 0.949999988079071
    2021-08-03 19:25:14,597 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 40, start read image
    2021-08-03 19:26:56,301 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:26:59,939 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:27:06,590 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 40, trainbatch 10, loss 0.21528372168540955, acc1 0.8500000238418579, time 0.67 sec
    2021-08-03 19:27:14,196 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 40, trainbatch 20, loss 0.19206053018569946, acc1 0.949999988079071, time 0.69 sec
    2021-08-03 19:27:14,225 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 41, start read image
    2021-08-03 19:29:13,857 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:29:19,390 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:29:23,705 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:29:27,050 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:29:31,100 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 41, trainbatch 10, loss 0.1926926076412201, acc1 0.949999988079071, time 0.72 sec
    2021-08-03 19:29:33,816 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:29:37,064 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:29:40,273 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:29:46,866 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 41, trainbatch 20, loss 0.20466457307338715, acc1 0.8999999761581421, time 0.87 sec
    2021-08-03 19:29:46,894 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 42, start read image
    2021-08-03 19:31:39,132 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:31:43,292 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:31:48,412 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:31:52,313 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 42, trainbatch 10, loss 0.38245782256126404, acc1 0.8500000238418579, time 0.81 sec
    2021-08-03 19:31:52,354 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 850 batch train result, current acc1 0.8500000238418579
    2021-08-03 19:31:56,489 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:32:01,217 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:32:03,575 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:32:07,789 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 42, trainbatch 20, loss 0.09285560995340347, acc1 0.949999988079071, time 0.64 sec
    2021-08-03 19:32:07,808 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 43, start read image
    2021-08-03 19:33:55,955 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:34:01,271 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:34:04,926 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:34:07,832 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:34:10,608 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 43, trainbatch 10, loss 0.19939617812633514, acc1 0.949999988079071, time 0.63 sec
    2021-08-03 19:34:11,914 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:34:16,311 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:34:19,711 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:34:26,728 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 43, trainbatch 20, loss 0.2334701120853424, acc1 0.8500000238418579, time 0.82 sec
    2021-08-03 19:34:26,752 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 44, start read image
    2021-08-03 19:36:27,730 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:36:31,024 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:36:34,689 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:36:37,673 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 44, trainbatch 10, loss 0.14090853929519653, acc1 0.8999999761581421, time 0.68 sec
    2021-08-03 19:36:38,355 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:36:43,018 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:36:45,631 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:36:48,817 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:36:53,841 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 44, trainbatch 20, loss 0.09580247849225998, acc1 0.949999988079071, time 0.68 sec
    2021-08-03 19:36:53,865 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 900 batch train result, current acc1 0.949999988079071
    2021-08-03 19:36:55,630 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 45, start read image
    2021-08-03 19:38:42,441 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:38:45,430 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:38:48,350 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:38:54,086 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:38:56,779 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 45, trainbatch 10, loss 0.07092894613742828, acc1 0.949999988079071, time 0.66 sec
    2021-08-03 19:39:01,279 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:39:04,863 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:39:07,266 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:39:10,115 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 45, trainbatch 20, loss 0.26860126852989197, acc1 0.8999999761581421, time 0.64 sec
    2021-08-03 19:39:10,132 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 46, start read image
    2021-08-03 19:40:57,087 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:41:00,640 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:41:04,473 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:41:09,303 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:41:11,663 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 46, trainbatch 10, loss 0.10878027975559235, acc1 0.949999988079071, time 0.64 sec
    2021-08-03 19:41:12,895 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:41:20,472 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:41:23,200 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 46, trainbatch 20, loss 0.24251563847064972, acc1 0.8999999761581421, time 0.67 sec
    2021-08-03 19:41:23,225 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 47, start read image
    2021-08-03 19:43:14,809 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:43:17,662 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:43:23,440 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:43:27,755 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 47, trainbatch 10, loss 0.1619100272655487, acc1 0.949999988079071, time 0.67 sec
    2021-08-03 19:43:27,777 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 950 batch train result, current acc1 0.949999988079071
    2021-08-03 19:43:32,484 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:43:36,743 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:43:40,972 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 47, trainbatch 20, loss 0.11219606548547745, acc1 0.949999988079071, time 1.23 sec
    2021-08-03 19:43:40,975 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 48, start read image
    2021-08-03 19:45:34,413 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 48, trainbatch 10, loss 0.09312238544225693, acc1 0.949999988079071, time 0.68 sec
    2021-08-03 19:45:35,076 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:45:38,544 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:45:42,093 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:45:45,936 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:45:50,177 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 48, trainbatch 20, loss 0.18698471784591675, acc1 0.8999999761581421, time 0.65 sec
    2021-08-03 19:45:50,200 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 49, start read image
    2021-08-03 19:47:40,447 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:47:42,971 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:47:46,937 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 49, trainbatch 10, loss 0.05294424667954445, acc1 1.0, time 0.68 sec
    2021-08-03 19:47:46,961 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:47:49,939 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:47:52,465 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:47:58,475 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:48:02,930 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:48:05,564 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 49, trainbatch 20, loss 0.3077161908149719, acc1 0.8999999761581421, time 0.70 sec
    2021-08-03 19:48:05,587 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 1000 batch train result, current acc1 0.8999999761581421
    2021-08-03 19:48:07,502 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 50, start read image
    2021-08-03 19:50:00,712 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:50:05,748 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:50:09,453 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:50:12,112 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 50, trainbatch 10, loss 0.1713104099035263, acc1 0.949999988079071, time 0.73 sec
    2021-08-03 19:50:13,368 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:50:17,819 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:50:20,312 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:50:24,030 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:50:27,268 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:50:29,817 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:50:32,868 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 50, trainbatch 20, loss 0.03817090764641762, acc1 1.0, time 0.69 sec
    2021-08-03 19:50:32,899 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 4
    2021-08-03 19:50:34,810 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 51, start read image
    2021-08-03 19:52:31,295 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:52:36,154 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:52:39,410 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:52:43,284 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:52:46,286 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 51, trainbatch 10, loss 0.2732970416545868, acc1 0.8999999761581421, time 0.67 sec
    2021-08-03 19:52:48,313 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:52:54,803 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:52:58,025 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 51, trainbatch 20, loss 0.10616544634103775, acc1 0.949999988079071, time 0.68 sec
    2021-08-03 19:52:58,053 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 52, start read image
    2021-08-03 19:54:52,207 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:54:55,311 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:55:01,279 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:55:03,733 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 52, trainbatch 10, loss 0.0520063079893589, acc1 1.0, time 0.68 sec
    2021-08-03 19:55:03,755 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:55:06,101 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 1050 batch train result, current acc1 1.0
    2021-08-03 19:55:08,459 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:55:12,142 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:55:19,212 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:55:21,856 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 52, trainbatch 20, loss 0.15372231602668762, acc1 0.949999988079071, time 0.76 sec
    2021-08-03 19:55:21,881 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 53, start read image
    2021-08-03 19:57:21,494 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:57:24,614 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:57:28,639 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:57:31,140 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:57:34,284 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:57:37,538 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 4
    2021-08-03 19:57:41,850 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:57:44,367 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 53, trainbatch 10, loss 0.01980108767747879, acc1 1.0, time 0.67 sec
    2021-08-03 19:57:44,388 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 19:57:47,518 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 19:57:50,218 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 4
    2021-08-03 19:57:56,571 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:58:01,758 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 53, trainbatch 20, loss 0.09003663808107376, acc1 1.0, time 0.68 sec
    2021-08-03 19:58:01,786 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 19:58:03,705 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 54, start read image
    2021-08-03 20:00:02,200 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 20:00:07,189 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 20:00:11,463 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 20:00:13,916 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 54, trainbatch 10, loss 0.07517781108617783, acc1 0.949999988079071, time 0.66 sec
    2021-08-03 20:00:15,786 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 20:00:23,378 - <ipython-input-18-f06b761b0d12>[line:63] - INFO: Pass 54, trainbatch 20, loss 0.05779242143034935, acc1 1.0, time 0.67 sec
    2021-08-03 20:00:23,401 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 20:00:25,570 - <ipython-input-18-f06b761b0d12>[line:82] - INFO: temp save 1100 batch train result, current acc1 1.0
    2021-08-03 20:00:27,287 - <ipython-input-18-f06b761b0d12>[line:48] - INFO: current pass: 55, start read image
    2021-08-03 20:02:10,864 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 1
    2021-08-03 20:02:13,881 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 2
    2021-08-03 20:02:17,100 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 3
    2021-08-03 20:02:19,460 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 4
    2021-08-03 20:02:22,375 - <ipython-input-18-f06b761b0d12>[line:67] - INFO: current acc1 1.0 meets good 0.96, successive count 5
    2021-08-03 20:02:24,143 - <ipython-input-18-f06b761b0d12>[line:74] - INFO: end training
    2021-08-03 20:02:24,151 - <ipython-input-18-f06b761b0d12>[line:88] - INFO: training till last epcho, end training



```python
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  

import os  
import numpy as np  
import random  
import time  
import codecs  
import sys  
import functools  
import math  
import paddle  
import paddle.fluid as fluid  
from paddle.fluid import core  
from paddle.fluid.param_attr import ParamAttr  
from PIL import Image, ImageEnhance  

paddle.enable_static()
target_size = [3, 512, 512]  
mean_rgb = [127.5, 127.5, 127.5]  
data_dir = ""  
eval_file = "valdata.txt"  
use_gpu = True  
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()  
exe = fluid.Executor(place)  
save_freeze_dir = "./freeze-model"  
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_freeze_dir, executor=exe)  
print(fetch_targets)  

def crop_image(img, target_size):  
    width, height = img.size
    p = min(target_size[2] / width, target_size[1] / height)
    resized_h = int(height * p)
    resized_w = int(width * p)
    img = img.resize((resized_w, resized_h), Image.BILINEAR)
    w_start = (resized_w - target_size[2]) / 2  
    h_start = (resized_h - target_size[1]) / 2  
    w_end = w_start + target_size[2]  
    h_end = h_start + target_size[1]  
    img = img.crop((w_start, h_start, w_end, h_end))  
    return img  


def resize_img(img, target_size):  
    ret = img.resize((target_size[1], target_size[2]), Image.BILINEAR)  
    return ret  


def read_image(img_path):  
    img = Image.open(img_path)  
    if img.mode != 'RGB':  
        img = img.convert('RGB')  
    # img = crop_image(img, target_size)
    img = resize_img(img, target_size)
    img = np.array(img).astype('float32')  
    img -= mean_rgb  
    img = img.transpose((2, 0, 1))  # HWC to CHW  
    img *= 0.007843  
    img = img[np.newaxis,:]  
    return img  


def infer(image_path):  
    tensor_img = read_image(image_path)  
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)  
    return np.argmax(label)  


def eval_all():  
    eval_file_path = os.path.join(data_dir, eval_file)  
    total_count = 0  
    right_count = 0  
    with codecs.open(eval_file_path, encoding='utf-8') as flist:  
        lines = [line.strip() for line in flist]  
        t1 = time.time()  
        for line in lines:  
            total_count += 1  
            parts = line.strip().split()  
            result = infer(parts[0])  
            print(parts[0]+"infer result:{0} answer:{1}".format(result, parts[1]))  
            if str(result) == parts[1]:  
                right_count += 1  
        period = time.time() - t1  
        print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(total_count, "%2.2f sec" % period, right_count / total_count))  



```

    [var save_infer_model/scale_0.tmp_172 : LOD_TENSOR.shape(-1, 2).dtype(float32).stop_gradient(False)]
    PALM-Validation400/V0001.jpginfer result:0 answer:0
    PALM-Validation400/V0002.jpginfer result:0 answer:1
    PALM-Validation400/V0003.jpginfer result:1 answer:1
    PALM-Validation400/V0004.jpginfer result:0 answer:0
    PALM-Validation400/V0005.jpginfer result:0 answer:0
    PALM-Validation400/V0006.jpginfer result:0 answer:0
    PALM-Validation400/V0007.jpginfer result:0 answer:0
    PALM-Validation400/V0008.jpginfer result:1 answer:1
    PALM-Validation400/V0009.jpginfer result:1 answer:0
    PALM-Validation400/V0010.jpginfer result:0 answer:0
    PALM-Validation400/V0011.jpginfer result:1 answer:1
    PALM-Validation400/V0012.jpginfer result:1 answer:1
    PALM-Validation400/V0013.jpginfer result:0 answer:1
    PALM-Validation400/V0014.jpginfer result:1 answer:1
    PALM-Validation400/V0015.jpginfer result:0 answer:0
    PALM-Validation400/V0016.jpginfer result:1 answer:1
    PALM-Validation400/V0017.jpginfer result:1 answer:1
    PALM-Validation400/V0018.jpginfer result:0 answer:0
    PALM-Validation400/V0019.jpginfer result:1 answer:1
    PALM-Validation400/V0020.jpginfer result:0 answer:0
    PALM-Validation400/V0021.jpginfer result:1 answer:1
    PALM-Validation400/V0022.jpginfer result:1 answer:1
    PALM-Validation400/V0023.jpginfer result:1 answer:1
    PALM-Validation400/V0024.jpginfer result:0 answer:0
    PALM-Validation400/V0025.jpginfer result:0 answer:0
    PALM-Validation400/V0026.jpginfer result:1 answer:1
    PALM-Validation400/V0027.jpginfer result:1 answer:1
    PALM-Validation400/V0028.jpginfer result:1 answer:1
    PALM-Validation400/V0029.jpginfer result:1 answer:1
    PALM-Validation400/V0030.jpginfer result:1 answer:1
    PALM-Validation400/V0031.jpginfer result:0 answer:0
    PALM-Validation400/V0032.jpginfer result:0 answer:0
    PALM-Validation400/V0033.jpginfer result:1 answer:1
    PALM-Validation400/V0034.jpginfer result:1 answer:1
    PALM-Validation400/V0035.jpginfer result:1 answer:1
    PALM-Validation400/V0036.jpginfer result:1 answer:1
    PALM-Validation400/V0037.jpginfer result:0 answer:0
    PALM-Validation400/V0038.jpginfer result:0 answer:0
    PALM-Validation400/V0039.jpginfer result:0 answer:0
    PALM-Validation400/V0040.jpginfer result:1 answer:1
    PALM-Validation400/V0041.jpginfer result:0 answer:0
    PALM-Validation400/V0042.jpginfer result:0 answer:0
    PALM-Validation400/V0043.jpginfer result:1 answer:1
    PALM-Validation400/V0044.jpginfer result:1 answer:1
    PALM-Validation400/V0045.jpginfer result:0 answer:0
    PALM-Validation400/V0046.jpginfer result:1 answer:1
    PALM-Validation400/V0047.jpginfer result:1 answer:1
    PALM-Validation400/V0048.jpginfer result:1 answer:1
    PALM-Validation400/V0049.jpginfer result:1 answer:1
    PALM-Validation400/V0050.jpginfer result:0 answer:0
    PALM-Validation400/V0051.jpginfer result:0 answer:0
    PALM-Validation400/V0052.jpginfer result:1 answer:1
    PALM-Validation400/V0053.jpginfer result:1 answer:1
    PALM-Validation400/V0054.jpginfer result:0 answer:0
    PALM-Validation400/V0055.jpginfer result:1 answer:1
    PALM-Validation400/V0056.jpginfer result:0 answer:0
    PALM-Validation400/V0057.jpginfer result:0 answer:0
    PALM-Validation400/V0058.jpginfer result:0 answer:0
    PALM-Validation400/V0059.jpginfer result:0 answer:0
    PALM-Validation400/V0060.jpginfer result:0 answer:0
    PALM-Validation400/V0061.jpginfer result:0 answer:0
    PALM-Validation400/V0062.jpginfer result:0 answer:1
    PALM-Validation400/V0063.jpginfer result:1 answer:1
    PALM-Validation400/V0064.jpginfer result:0 answer:0
    PALM-Validation400/V0065.jpginfer result:0 answer:1
    PALM-Validation400/V0066.jpginfer result:1 answer:1
    PALM-Validation400/V0067.jpginfer result:1 answer:1
    PALM-Validation400/V0068.jpginfer result:1 answer:1
    PALM-Validation400/V0069.jpginfer result:0 answer:0



```python
if __name__ == '__main__':  
    eval_all()
```

## 6. 模型推理
### 设计一个接口函数，通过这个接口函数能够方便地对任意一个样本进行实时预测
imgpath为要预测的文件路径，直接调用infer_img(imgpath)即可实时预测


```python
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  

import os  
import numpy as np  
import random  
import time  
import codecs  
import sys  
import functools  
import math  
import paddle  
import paddle.fluid as fluid  
from paddle.fluid import core  
from paddle.fluid.param_attr import ParamAttr  
from PIL import Image, ImageEnhance  
def infer_img(imgpath):
    paddle.enable_static()
    target_size = [3, 512, 512]  
    mean_rgb = [127.5, 127.5, 127.5]
    use_gpu = True  
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()  
    exe = fluid.Executor(place)  
    save_freeze_dir = "./freeze-model4"
    labellist = ['high myopia or non-PM', 'PM']
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_freeze_dir, executor=exe)

    def crop_image(img, target_size):  
        width, height = img.size
        p = min(target_size[2] / width, target_size[1] / height)
        resized_h = int(height * p)
        resized_w = int(width * p)
        img = img.resize((resized_w, resized_h), Image.BILINEAR)
        w_start = (resized_w - target_size[2]) / 2  
        h_start = (resized_h - target_size[1]) / 2  
        w_end = w_start + target_size[2]  
        h_end = h_start + target_size[1]  
        img = img.crop((w_start, h_start, w_end, h_end))  
        return img  


    def resize_img(img, target_size):  
        ret = img.resize((target_size[1], target_size[2]), Image.BILINEAR)  
        return ret  


    def read_image(img_path):  
        img = Image.open(img_path)  
        if img.mode != 'RGB':  
            img = img.convert('RGB')  
        # img = crop_image(img, target_size)
        img = resize_img(img, target_size)
        img = np.array(img).astype('float32')  
        img -= mean_rgb  
        img = img.transpose((2, 0, 1))  # HWC to CHW  
        img *= 0.007843  
        img = img[np.newaxis,:]  
        return img  


    def infer(image_path):  
        tensor_img = read_image(image_path)  
        label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)  
        return np.argmax(label)  


    def eval_one(imgpath):
        result = infer(imgpath)  
        print(imgpath + "------infer result:{0}".format(labellist[result]))

    return eval_one(imgpath)
```


```python
imgpath = 'PALM-Training400/P0009.jpg'
infer_img(imgpath)
```

    PALM-Training400/P0009.jpg------infer result:PM



```python
#打包模型文件
!tar -cf persis_densenet4.tar ./persistable-params4
```
