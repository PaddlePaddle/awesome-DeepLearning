数据集请自行下载：

https://www.kaggle.com/c/criteo-display-ad-challenge/

# DeepFM

* 简单介绍
* 架构说明

## 简单介绍

​	目前大部分的CTR预估模型 “are biased to low- or high- order feature interaction”，例如FNN、PNN等NN模型专注于隐式的高阶特征相关性，而LR、FM等则专注于显式的低阶特征相关性，Goggle在2016年提出的Wide & Deep模型同时考虑了两者，但Wide部分需要人工参与的特征工程。论文动机非常直观，既希望考虑高/低阶的feature interaction，又想省去额外的特征工程。使用FM取代Wide的LR部分是一个可行的做法，当然这里LR 可以基于先验构造更高阶的组合特征，而FM只考虑二阶。

## 架构说明

DeepFM的网络结构如下图所示：![deepfm](C:\Users\SongWood\shujiaxuexi\baidushixi\DeepFM_torch-master\images\deepfm.jpg)

​	仔细观察上图，DeepFM 的结构其实很简单：**采取Wide & Deep的框架**，差异在于将Wide部分的LR替换为了**FM**，从而自动构造二阶特征叉乘，而非手工设计叉乘。

![deepfm2](C:\Users\SongWood\shujiaxuexi\baidushixi\DeepFM_torch-master\images\deepfm2.jpg)

​	将图1的Wide部分单独拉出来，即标准的FM结构，如上图2所示。值得注意的一点是，FM层与NN层 **share the same feature embedding**，而非各自学习各自部分的embedding。这么做的好处是：1）降低模型复杂度；2）在embedding的学习中同时接收与来自“low & high order interaction”部分的反馈，从而学到更好的特征表示。