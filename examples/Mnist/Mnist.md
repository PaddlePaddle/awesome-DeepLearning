```python

```

# 深度学习基础知识
### 损失函数
<font face="黑体" size = 2>
交叉熵，其用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小,也就是两个概率分布之间的距离.给定两个概率分布p和q，通过q来表示p的交叉熵就是通过概率分布q来表达概率分布p的困难程度，p代表正确答案，q代表的是预测值，交叉熵越小，两个概率的分布约接近。


```python
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size) # ndarray的size属性是存在的
		y = y.reshape(1, y.size)
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y+ 1e-7)) / batch_size

```

### 池化方法
<font face="黑体" size = 2>
空间金字塔池化可以把任何尺度的图像的卷积特征转化成相同维度，这不仅可以让CNN处理任意尺度的图像，还能避免cropping和warping操作，导致一些信息的丢失，具有非常重要的意义。

一般的CNN都需要输入图像的大小是固定的，这是因为全连接层的输入需要固定输入维度，但在卷积操作是没有对图像尺度有限制，所有作者提出了空间金字塔池化，先让图像进行卷积操作，然后转化成维度相同的特征输入到全连接层，这个可以把CNN扩展到任意大小的图像。

空间金字塔池化的思想来自于Spatial Pyramid Model，它一个pooling变成了多个scale的pooling。用不同大小池化窗口作用于卷积特征

![avatar](/1.png)


```python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

def spp_layer(input_, levels=4, name = 'SPP_layer',pool_type = 'max_pool'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):

        for l in range(levels):
        #设置池化参数
            l = l + 1
            ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]

            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1),)

            else :
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1：
                x_flatten = tf.reshape(pool,(shape[0],-1))
            else:
                x_flatten = tf.concat((x_flatten,pool),axis=1) #四种尺度进行拼接
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
    return x_flatten
```

### 数据增强方法
<font face="黑体" size = 2>
 1、平移。在图像平面上对图像以一定方式进行平移。

2、翻转图像。沿着水平或者垂直方向翻转图像。

3、旋转角度。随机旋转图像一定角度; 改变图像内容的朝向。

4、随机颜色。对图像进行颜色抖动，对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声。

5、对比度增强。增强图像对比度，也可以用直方图均衡化。

6、亮度增强。将整个图像亮度调高。

7、颜色增强。

8、还有随机裁剪、尺度变换等代码就不赘述了。


```python
# 平移
from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np
def move(root_path,img_name,off): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = img.offset(off,0)
    return offset
```

### 图像分类方法
<font face="黑体" size = 2>
第一类方法：使用 KNN、SVM、BP 神经网络这些课堂算法。这些算法强大易实现。我们主要使用 sklearn 实现这些算法。

第二类方法：尽管传统的多层感知器模型已成功应用于图像识别，但由于其节点之间的全连接性，它们遭遇了维度的难题，从而不能很好地扩展到更高分辨率的图像。因此我们使用深度学习框架 TensorFlow 打造了一个 CNN。

第三个方法：重新训练一个被称作 Inception V3 的预训练深度神经网络的最后一层，同样由 TensorFlow 提供。Inception V3 是为 ImageNet 大型视觉识别挑战赛训练的，使用了 2012 年的数据。这是计算机视觉的常规任务，其中模型试图把全部图像分为 1000 个类别，比如斑马、达尔阿提亚人和洗碗机。为了再训练这一预训练网络，我们要保证自己的数据集没有被预训练。

