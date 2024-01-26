import cv2
import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
import math

DATADIR = "./data/PALM-Training400/PALM-Training400/"
DATADIR2 = "./data/PALM-Validation400/"
CSVFILE = "./data/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv"


def transform_img(img):
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype("float32")
    img = img / 255.0
    img = img * 2.0 - 1.0
    return img


def data_loader(datadir, batch_size=10, mode="train"):
    filenames = os.listdir(datadir)

    def reader():
        if mode == "train":
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == "H" or name[0] == "N":
                label = 0
            elif name[0] == "P":
                label = 1
            else:
                raise ("Not excepted file name")
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype("float32")
                labels_array = np.array(batch_labels).astype("float32").reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype("float32")
            labels_array = np.array(batch_labels).astype("float32").reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


def valid_data_loader(datadir, csvfile, batch_size=10, mode="valid"):

    filelists = open(csvfile).readlines()

    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(",")
            name = line[1]
            label = int(line[2])
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype("float32")
                labels_array = np.array(batch_labels).astype("float32").reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype("float32")
            labels_array = np.array(batch_labels).astype("float32").reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


class ResNet50(fluid.dygraph.Layer):
    def __init__(self, class_dim=1):
        super(ResNet50, self).__init__()
        # self.conv = ConvBNLayer(num_channels=3,num_filters=64,filter_size=7,stride=2,act='relu')
        self.conv = Conv2D(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            padding=3,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm = BatchNorm(64, act="relu")
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type="max"
        )

        ###第一个模块里面有三个bottleneck_block
        # 前两个数字代表 第一个模块里面的第一个bottleneck_block
        ###bottleneck_block = 'bb_0_0',BottleneckBlock(num_channels=64,num_filters=num_filters[0]=64,stride=1,shortcut=False)
        # self.conv0 = ConvBNLayer(num_channels=64, num_filters=64,filter_size=1,act='relu')
        self.conv1_1_1_ = Conv2D(
            num_channels=64,
            num_filters=64,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_1_1_ = BatchNorm(64, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=64, num_filters=64,filter_size=3,stride=1,act='relu')
        self.conv1_1_2_ = Conv2D(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_1_2_ = BatchNorm(64, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=64, num_filters=64*4=256,filter_size=1,act=None)
        self.conv1_1_3_ = Conv2D(
            num_channels=64,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_1_3_ = BatchNorm(256, act=None)
        # shortcut = False
        # self.short = ConvBNLayer(num_channels=64, num_filters=64*4=256,filter_size=1,stride=1)
        self.conv1_1_4_ = Conv2D(
            num_channels=64,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_1_4_ = BatchNorm(256, act=None)
        # self.shortcut = False

        ###bottleneck_block = 'bb_0_1',BottleneckBlock(num_channels=256,num_filters=num_filters[0]=64,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=256, num_filters=64,filter_size=1,act='relu')
        self.conv1_2_1_ = Conv2D(
            num_channels=256,
            num_filters=64,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_2_1_ = BatchNorm(64, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=64, num_filters=64,filter_size=3,stride=1,act='relu')
        self.conv1_2_2_ = Conv2D(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_2_2_ = BatchNorm(64, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=64, num_filters=64*4=256,filter_size=1,act=None)
        self.conv1_2_3_ = Conv2D(
            num_channels=64,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_2_3_ = BatchNorm(256, act=None)
        # shortcut = True
        # self.shortcut = True

        ###bottleneck_block = 'bb_0_2',BottleneckBlock(num_channels=256,num_filters=num_filters[0]=64,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=256, num_filters=64,filter_size=1,act='relu')
        self.conv1_3_1_ = Conv2D(
            num_channels=256,
            num_filters=64,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_3_1_ = BatchNorm(64, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=64, num_filters=64,filter_size=3,stride=1,act='relu')
        self.conv1_3_2_ = Conv2D(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_3_2_ = BatchNorm(64, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=64, num_filters=64*4=256,filter_size=1,act=None)
        self.conv1_3_3_ = Conv2D(
            num_channels=64,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm1_3_3_ = BatchNorm(256, act=None)
        # shortcut = True
        # self.shortcut = True

        ##################################     第二个模块
        ###bottleneck_block = 'bb_1_0',BottleneckBlock(num_channels=256,num_filters=num_filters[1]=128,stride=2,shortcut=False)   #实例化BottleneckBlock的上面没有写
        # self.conv0 = ConvBNLayer(num_channels=256, num_filters=128,filter_size=1,act='relu')
        self.conv2_1_1_ = Conv2D(
            num_channels=256,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_1_1_ = BatchNorm(128, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=128, num_filters=128,filter_size=3,stride=2,act='relu')
        self.conv2_1_2_ = Conv2D(
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride=2,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_1_2_ = BatchNorm(128, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=128, num_filters=128*4=512,filter_size=1,act=None)
        self.conv2_1_3_ = Conv2D(
            num_channels=128,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_1_3_ = BatchNorm(512, act=None)
        # shortcut = False
        # self.short = ConvBNLayer(num_channels=256, num_filters=128*4=512,filter_size=1,stride=2)
        #####ResNet-D
        self.pool_2 = Pool2D(pool_size=2, pool_stride=2, pool_type="avg")

        self.conv2_1_4_ = Conv2D(
            num_channels=256,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_1_4_ = BatchNorm(512, act=None)
        # self.shortcut = False

        ###bottleneck_block = 'bb_1_1',BottleneckBlock(num_channels=512,num_filters=num_filters[1]=128,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=512, num_filters=128,filter_size=1,act='relu')
        self.conv2_2_1_ = Conv2D(
            num_channels=512,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_2_1_ = BatchNorm(128, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=128, num_filters=128,filter_size=3,stride=1,act='relu')
        self.conv2_2_2_ = Conv2D(
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_2_2_ = BatchNorm(128, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=128, num_filters=128*4=512,filter_size=1,act=None)
        self.conv2_2_3_ = Conv2D(
            num_channels=128,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_2_3_ = BatchNorm(512, act=None)
        # shortcut = True
        # self.shortcut = True

        ###bottleneck_block = 'bb_1_2',BottleneckBlock(num_channels=512,num_filters=num_filters[1]=128,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=512, num_filters=128,filter_size=1,act='relu')
        self.conv2_3_1_ = Conv2D(
            num_channels=512,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_3_1_ = BatchNorm(128, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=128, num_filters=128,filter_size=3,stride=1,act='relu')
        self.conv2_3_2_ = Conv2D(
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_3_2_ = BatchNorm(128, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=128, num_filters=128*4=512,filter_size=1,act=None)
        self.conv2_3_3_ = Conv2D(
            num_channels=128,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_3_3_ = BatchNorm(512, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_1_3',BottleneckBlock(num_channels=512,num_filters=num_filters[1]=128,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=512, num_filters=128,filter_size=1,act='relu')
        self.conv2_4_1_ = Conv2D(
            num_channels=512,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_4_1_ = BatchNorm(128, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=128, num_filters=128,filter_size=3,stride=1,act='relu')
        self.conv2_4_2_ = Conv2D(
            num_channels=128,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_4_2_ = BatchNorm(128, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=128, num_filters=128*4=512,filter_size=1,act=None)
        self.conv2_4_3_ = Conv2D(
            num_channels=128,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm2_4_3_ = BatchNorm(512, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_2_0',BottleneckBlock(num_channels=512,num_filters=num_filters[2]=256,stride=2,shortcut=False)
        # self.conv0 = ConvBNLayer(num_channels=512, num_filters=256,filter_size=1,act='relu')
        self.conv3_1_1_ = Conv2D(
            num_channels=512,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_1_1_ = BatchNorm(256, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=256, num_filters=256,filter_size=3,stride=2,act='relu')
        self.conv3_1_2_ = Conv2D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=2,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_1_2_ = BatchNorm(256, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=256, num_filters=256*4=1024,filter_size=1,act=None)
        self.conv3_1_3_ = Conv2D(
            num_channels=256,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_1_3_ = BatchNorm(1024, act=None)
        # shortcut = False
        # self.short = ConvBNLayer(num_channels=512, num_filters=256*4=1024,filter_size=1,stride=2)
        self.pool_3 = Pool2D(pool_size=2, pool_stride=2, pool_type="avg")
        self.conv3_1_4_ = Conv2D(
            num_channels=512,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_1_4_ = BatchNorm(1024, act=None)
        # self.shortcut = False

        # bottleneck_block = 'bb_2_1',BottleneckBlock(num_channels=1024,num_filters=num_filters[2]=256,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=1024, num_filters=256,filter_size=1,act='relu')
        self.conv3_2_1_ = Conv2D(
            num_channels=1024,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_2_1_ = BatchNorm(256, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=256, num_filters=256,filter_size=3,stride=1,act='relu')
        self.conv3_2_2_ = Conv2D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_2_2_ = BatchNorm(256, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=256, num_filters=256*4=1024,filter_size=1,act=None)
        self.conv3_2_3_ = Conv2D(
            num_channels=256,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_2_3_ = BatchNorm(1024, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_2_2',BottleneckBlock(num_channels=1024,num_filters=num_filters[2]=256,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=1024, num_filters=256,filter_size=1,act='relu')
        self.conv3_3_1_ = Conv2D(
            num_channels=1024,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_3_1_ = BatchNorm(256, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=256, num_filters=256,filter_size=3,stride=1,act='relu')
        self.conv3_3_2_ = Conv2D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_3_2_ = BatchNorm(256, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=256, num_filters=256*4=1024,filter_size=1,act=None)
        self.conv3_3_3_ = Conv2D(
            num_channels=256,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_3_3_ = BatchNorm(1024, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_2_3',BottleneckBlock(num_channels=1024,num_filters=num_filters[2]=256,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=1024, num_filters=256,filter_size=1,act='relu')
        self.conv3_4_1_ = Conv2D(
            num_channels=1024,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_4_1_ = BatchNorm(256, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=256, num_filters=256,filter_size=3,stride=1,act='relu')
        self.conv3_4_2_ = Conv2D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_4_2_ = BatchNorm(256, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=256, num_filters=256*4=1024,filter_size=1,act=None)
        self.conv3_4_3_ = Conv2D(
            num_channels=256,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_4_3_ = BatchNorm(1024, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_2_4',BottleneckBlock(num_channels=1024,num_filters=num_filters[2]=256,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=1024, num_filters=256,filter_size=1,act='relu')
        self.conv3_5_1_ = Conv2D(
            num_channels=1024,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_5_1_ = BatchNorm(256, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=256, num_filters=256,filter_size=3,stride=1,act='relu')
        self.conv3_5_2_ = Conv2D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_5_2_ = BatchNorm(256, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=256, num_filters=256*4=1024,filter_size=1,act=None)
        self.conv3_5_3_ = Conv2D(
            num_channels=256,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_5_3_ = BatchNorm(1024, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_2_5',BottleneckBlock(num_channels=1024,num_filters=num_filters[2]=256,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=1024, num_filters=256,filter_size=1,act='relu')
        self.conv3_6_1_ = Conv2D(
            num_channels=1024,
            num_filters=256,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_6_1_ = BatchNorm(256, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=256, num_filters=256,filter_size=3,stride=1,act='relu')
        self.conv3_6_2_ = Conv2D(
            num_channels=256,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_6_2_ = BatchNorm(256, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=256, num_filters=256*4=1024,filter_size=1,act=None)
        self.conv3_6_3_ = Conv2D(
            num_channels=256,
            num_filters=1024,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm3_6_3_ = BatchNorm(1024, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_3_0',BottleneckBlock(num_channels=1024,num_filters=num_filters[3]=512,stride=2,shortcut=False)
        # self.conv0 = ConvBNLayer(num_channels=1024, num_filters=512,filter_size=1,act='relu')
        self.conv4_1_1_ = Conv2D(
            num_channels=1024,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_1_1_ = BatchNorm(512, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=512, num_filters=512,filter_size=3,stride=2,act='relu')
        self.conv4_1_2_ = Conv2D(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride=2,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_1_2_ = BatchNorm(512, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=512, num_filters=512*4=2048,filter_size=1,act=None)
        self.conv4_1_3_ = Conv2D(
            num_channels=512,
            num_filters=2048,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_1_3_ = BatchNorm(2048, act=None)
        # shortcut = False
        # self.short = ConvBNLayer(num_channels=1024, num_filters=512*4=2048,filter_size=1,stride=2)
        self.pool_4 = Pool2D(pool_size=2, pool_stride=2, pool_type="avg")
        self.conv4_1_4_ = Conv2D(
            num_channels=1024,
            num_filters=2048,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_1_4_ = BatchNorm(2048, act=None)
        # self.shortcut = False

        # bottleneck_block = 'bb_3_1',BottleneckBlock(num_channels=2048,num_filters=num_filters[3]=512,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=2048, num_filters=512,filter_size=1,act='relu')
        self.conv4_2_1_ = Conv2D(
            num_channels=2048,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_2_1_ = BatchNorm(512, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=512, num_filters=512,filter_size=3,stride=1,act='relu')
        self.conv4_2_2_ = Conv2D(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_2_2_ = BatchNorm(512, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=512, num_filters=512*4=2048,filter_size=1,act=None)
        self.conv4_2_3_ = Conv2D(
            num_channels=512,
            num_filters=2048,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_2_3_ = BatchNorm(2048, act=None)
        # shortcut = True
        # self.shortcut = True

        # bottleneck_block = 'bb_3_2',BottleneckBlock(num_channels=2048,num_filters=num_filters[3]=512,stride=1,shortcut=True)
        # self.conv0 = ConvBNLayer(num_channels=2048, num_filters=512,filter_size=1,act='relu')
        self.conv4_3_1_ = Conv2D(
            num_channels=2048,
            num_filters=512,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_3_1_ = BatchNorm(512, act="relu")
        # self.conv1 = ConvBNLayer(num_channels=512, num_filters=512,filter_size=3,stride=1,act='relu')
        self.conv4_3_2_ = Conv2D(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_3_2_ = BatchNorm(512, act="relu")
        # self.conv2 = ConvBNLayer(num_channels=512, num_filters=512*4=2048,filter_size=1,act=None)
        self.conv4_3_3_ = Conv2D(
            num_channels=512,
            num_filters=2048,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            bias_attr=False,
        )
        self.batch_norm4_3_3_ = BatchNorm(2048, act=None)
        # shortcut = True
        # self.shortcut = True

        self.pool2d_avg = Pool2D(pool_size=7, pool_type="avg", global_pooling=True)
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.out = Linear(
            input_dim=2048,
            output_dim=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)
            ),
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.pool2d_max(x)

        inputs1_1 = x  # 这是第一个模块中第一个buttonneck的输入
        x = self.conv1_1_1_(inputs1_1)
        x = self.batch_norm1_1_1_(x)  # 得到的x 是y

        x = self.conv1_1_2_(x)
        x = self.batch_norm1_1_2_(x)  # 得到的x是 conv1

        x = self.conv1_1_3_(x)
        conv1_1 = self.batch_norm1_1_3_(x)  # 得到的x是conv2
        ####这里要对shortcut做判断
        x = self.conv1_1_4_(inputs1_1)
        short1_1 = self.batch_norm1_1_4_(x)
        # 这是第一个模块中第一个buttonneck的输出
        x = fluid.layers.elementwise_add(x=short1_1, y=conv1_1, act="relu")

        inputs1_2 = x
        x = self.conv1_2_1_(inputs1_2)
        x = self.batch_norm1_2_1_(x)

        x = self.conv1_2_2_(x)
        x = self.batch_norm1_2_2_(x)

        x = self.conv1_2_3_(x)
        conv1_2 = self.batch_norm1_2_3_(x)

        short1_2 = inputs1_2

        x = fluid.layers.elementwise_add(x=short1_2, y=conv1_2, act="relu")

        inputs1_3 = x
        x = self.conv1_3_1_(inputs1_3)
        x = self.batch_norm1_3_1_(x)

        x = self.conv1_3_2_(x)
        x = self.batch_norm1_3_2_(x)

        x = self.conv1_3_3_(x)
        conv1_3 = self.batch_norm1_3_3_(x)

        short1_3 = inputs1_3

        x = fluid.layers.elementwise_add(x=short1_3, y=conv1_3, act="relu")
        #################################第二个模块
        inputs2_1 = x
        x = self.conv2_1_1_(inputs2_1)
        x = self.batch_norm2_1_1_(x)

        x = self.conv2_1_2_(x)
        x = self.batch_norm2_1_2_(x)

        x = self.conv2_1_3_(x)
        conv2_1 = self.batch_norm2_1_3_(x)

        # self.pool_2 = Pool2D(pool_size=2,pool_stride=2,pool_type='avg')
        x = self.pool_2(inputs2_1)
        x = self.conv2_1_4_(x)
        short2_1 = self.batch_norm2_1_4_(x)

        x = fluid.layers.elementwise_add(x=short2_1, y=conv2_1, act="relu")

        inputs2_2 = x
        x = self.conv2_2_1_(inputs2_2)
        x = self.batch_norm2_2_1_(x)

        x = self.conv2_2_2_(x)
        x = self.batch_norm2_2_2_(x)

        x = self.conv2_2_3_(x)
        conv2_2 = self.batch_norm2_2_3_(x)

        short2_2 = inputs2_2

        x = fluid.layers.elementwise_add(x=short2_2, y=conv2_2, act="relu")

        inputs2_3 = x
        x = self.conv2_3_1_(inputs2_3)
        x = self.batch_norm2_3_1_(x)

        x = self.conv2_3_2_(x)
        x = self.batch_norm2_3_2_(x)

        x = self.conv2_3_3_(x)
        conv2_3 = self.batch_norm2_3_3_(x)

        short2_3 = inputs2_3

        x = fluid.layers.elementwise_add(x=short2_3, y=conv2_3, act="relu")

        inputs2_4 = x
        x = self.conv2_4_1_(inputs2_4)
        x = self.batch_norm2_4_1_(x)

        x = self.conv2_4_2_(x)
        x = self.batch_norm2_4_2_(x)

        x = self.conv2_4_3_(x)
        conv2_4 = self.batch_norm2_4_3_(x)

        short2_4 = inputs2_4

        x = fluid.layers.elementwise_add(x=short2_4, y=conv2_4, act="relu")

        ######################第三个模块
        inputs3_1 = x
        x = self.conv3_1_1_(inputs3_1)
        x = self.batch_norm3_1_1_(x)

        x = self.conv3_1_2_(x)
        x = self.batch_norm3_1_2_(x)

        x = self.conv3_1_3_(x)
        conv3_1 = self.batch_norm3_1_3_(x)

        # self.pool_2
        x = self.pool_3(inputs3_1)
        x = self.conv3_1_4_(x)
        short3_1 = self.batch_norm3_1_4_(x)

        x = fluid.layers.elementwise_add(x=short3_1, y=conv3_1, act="relu")

        inputs3_2 = x
        x = self.conv3_2_1_(inputs3_2)
        x = self.batch_norm3_2_1_(x)

        x = self.conv3_2_2_(x)
        x = self.batch_norm3_2_2_(x)

        x = self.conv3_2_3_(x)
        conv3_2 = self.batch_norm3_2_3_(x)

        short3_2 = inputs3_2

        x = fluid.layers.elementwise_add(x=short3_2, y=conv3_2, act="relu")

        inputs3_3 = x
        x = self.conv3_3_1_(inputs3_3)
        x = self.batch_norm3_3_1_(x)

        x = self.conv3_3_2_(x)
        x = self.batch_norm3_3_2_(x)

        x = self.conv3_3_3_(x)
        conv3_3 = self.batch_norm3_3_3_(x)

        short3_3 = inputs3_3

        x = fluid.layers.elementwise_add(x=short3_3, y=conv3_3, act="relu")

        inputs3_4 = x
        x = self.conv3_4_1_(inputs3_4)
        x = self.batch_norm3_4_1_(x)

        x = self.conv3_4_2_(x)
        x = self.batch_norm3_4_2_(x)

        x = self.conv3_4_3_(x)
        conv3_4 = self.batch_norm3_4_3_(x)

        short3_4 = inputs3_4

        x = fluid.layers.elementwise_add(x=short3_4, y=conv3_4, act="relu")

        inputs3_5 = x
        x = self.conv3_5_1_(inputs3_5)
        x = self.batch_norm3_5_1_(x)

        x = self.conv3_5_2_(x)
        x = self.batch_norm3_5_2_(x)

        x = self.conv3_5_3_(x)
        conv3_5 = self.batch_norm3_5_3_(x)

        short3_5 = inputs3_5

        x = fluid.layers.elementwise_add(x=short3_5, y=conv3_5, act="relu")

        inputs3_6 = x
        x = self.conv3_6_1_(inputs3_6)
        x = self.batch_norm3_6_1_(x)

        x = self.conv3_6_2_(x)
        x = self.batch_norm3_6_2_(x)

        x = self.conv3_6_3_(x)
        conv3_6 = self.batch_norm3_6_3_(x)

        short3_6 = inputs3_6

        x = fluid.layers.elementwise_add(x=short3_6, y=conv3_6, act="relu")

        #############################第4个模块
        inputs4_1 = x
        x = self.conv4_1_1_(inputs4_1)
        x = self.batch_norm4_1_1_(x)

        x = self.conv4_1_2_(x)
        x = self.batch_norm4_1_2_(x)

        x = self.conv4_1_3_(x)
        conv4_1 = self.batch_norm4_1_3_(x)

        # self.pool_2
        x = self.pool_4(inputs4_1)
        x = self.conv4_1_4_(x)
        short4_1 = self.batch_norm4_1_4_(x)

        x = fluid.layers.elementwise_add(x=short4_1, y=conv4_1, act="relu")

        inputs4_2 = x
        x = self.conv4_2_1_(inputs4_2)
        x = self.batch_norm4_2_1_(x)

        x = self.conv4_2_2_(x)
        x = self.batch_norm4_2_2_(x)

        x = self.conv4_2_3_(x)
        conv4_2 = self.batch_norm4_2_3_(x)

        short4_2 = inputs4_2

        x = fluid.layers.elementwise_add(x=short4_2, y=conv4_2, act="relu")

        inputs4_3 = x
        x = self.conv4_3_1_(inputs4_3)
        x = self.batch_norm4_3_1_(x)

        x = self.conv4_3_2_(x)
        x = self.batch_norm4_3_2_(x)

        x = self.conv4_3_3_(x)
        conv4_3 = self.batch_norm4_3_3_(x)

        short4_3 = inputs4_3

        x = fluid.layers.elementwise_add(x=short4_3, y=conv4_3, act="relu")

        x = self.pool2d_avg(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])  # [10,2048]
        y = self.out(x)  # [10,1]
        return y


with fluid.dygraph.guard():
    model = ResNet50()
    print("start training ... ")
    model.train()
    epoch_num = 1
    # 定义优化器
    opt = fluid.optimizer.Momentum(
        learning_rate=0.001, momentum=0.9, parameter_list=model.parameters()
    )
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batch_size=10, mode="train")
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            logits = model(img)
            # 进行loss计算
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                logits, label
            )  # loss的目的是让sigmoid(logits)去逼近label 所以在预测的时候预测值是sigmoid(logits)
            avg_loss = fluid.layers.mean(loss)
            if batch_id % 10 == 0:
                print(
                    "epoch: {}, batch_id: {}, loss is: {}".format(
                        epoch, batch_id, avg_loss.numpy()
                    )
                )
            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(img)
            # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
            # 计算sigmoid后的预测概率，进行loss计算
            pred = fluid.layers.sigmoid(logits)  ## 这个值大余）0.5就代表预测值为1
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
            pred2 = pred * (-1.0) + 1.0
            # 得到两个类别的预测概率，并沿第一个维度级联
            pred = fluid.layers.concat([pred2, pred], axis=1)  # [10，2]
            acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype="int64"))
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print(
            "[validation] accuracy/loss: {}/{}".format(
                np.mean(accuracies), np.mean(losses)
            )
        )
        model.train()
    # save params of model
    fluid.save_dygraph(model.state_dict(), "palm")
    # save optimizer state
    fluid.save_dygraph(opt.state_dict(), "palm")
