YouTube深度学习视频推荐系统

概念

作为全球最大的视频分享网站，YouTube 平台中几乎所有的视频都来自 UGC（User Generated Content，用户原创内容），这样的内容产生模式有两个特点：

一是其商业模式不同于 Netflix，以及国内的腾讯视频、爱奇艺这样的流媒体，这些流媒体的大部分内容都是采购或自制的电影、剧集等头部内容，而 YouTube 的内容都是用户上传的自制视频，种类风格繁多，头部效应没那么明显；
二是由于 YouTube 的视频基数巨大，用户难以发现喜欢的内容。

为了对海量的视频进行快速、准确的排序，YouTube 采用了经典的召回层 + 排序层的推荐系统架构，被称为YouTube深度学习视频推荐系统。

原理和流程

![aHR0cHM6Ly9waWMyLnpoaW1nLmNvbS84MC92Mi1kZDljMTQyOGEzMDQzMzAzNjEzMzFhOTI0ZjJjMzlmZF9oZC5qcGc](C:\Users\apple\Desktop\image\aHR0cHM6Ly9waWMyLnpoaW1nLmNvbS84MC92Mi1kZDljMTQyOGEzMDQzMzAzNjEzMzFhOTI0ZjJjMzlmZF9oZC5qcGc.jpg)

Youtube作为全球最大的UGC的视频网站，需要在百万量级的视频规模下进行个性化推荐。由于候选视频集合过大，考虑online系统延迟问题，不宜用复杂网络直接进行推荐，所以Youtube采取了两层深度网络完成整个推荐过程：

1. 第一层是Candidate Generation Model完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。
2. 第二层是用Ranking Model完成几百个候选视频的精排

Candidate Generation Model架构：

![aHR0cHM6Ly9waWMyLnpoaW1nLmNvbS84MC92Mi1iNjk5OGMzMWExYjI5MTI1ZDhkYTQwM2I5YTdiMjliZF9oZC5qcGc](C:\Users\apple\Desktop\image\aHR0cHM6Ly9waWMyLnpoaW1nLmNvbS84MC92Mi1iNjk5OGMzMWExYjI5MTI1ZDhkYTQwM2I5YTdiMjliZF9oZC5qcGc.jpg)

最底层是它的输入层，输入的特征包括用户历史观看视频的 Embedding 向量，以及搜索词的 Embedding 向量。对于这些 Embedding 特征，YouTube 是利用用户的观看序列和搜索序列，采用了类似 Item2vec 的预训练方式生成的。

除了视频和搜索词 Embedding 向量，特征向量中还包括用户的地理位置 Embedding、年龄、性别等特征。这里我们需要注意的是，对于样本年龄这个特征，YouTube 不仅使用了原始特征值，还把经过平方处理的特征值也作为一个新的特征输入模型。
这个操作其实是为了挖掘特征非线性的特性。

确定好了特征，这些特征会在 concat 层中连接起来，输入到上层的 ReLU 神经网络进行训练。

三层 ReLU 神经网络过后，YouTube 又使用了 softmax 函数作为输出层。值得一提的是，这里的输出层不是要预测用户会不会点击这个视频，而是要预测用户会点击哪个视频，这就跟一般深度推荐模型不一样。

总的来讲，YouTube 推荐系统的候选集生成模型，是一个标准的利用了 Embedding 预训练特征的深度推荐模型，它遵循Embedding MLP 模型的架构，只是在最后的输出层有所区别。

Ranking Model架构：

![aHR0cHM6Ly9waWMxLnpoaW1nLmNvbS84MC92Mi1iZDgyNGZlNmMxZjViZWIxZDFlYjg5Njk1NTczNGZkY19oZC5qcGc](C:\Users\apple\Desktop\image\aHR0cHM6Ly9waWMxLnpoaW1nLmNvbS84MC92Mi1iZDgyNGZlNmMxZjViZWIxZDFlYjg5Njk1NTczNGZkY19oZC5qcGc.jpg)

输入层，相比于候选集生成模型需要对几百万候选集进行粗筛，排序模型只需对几百个候选视频进行排序，因此可以引入更多特征进行精排。具体来说，YouTube 的输入层从左至右引入的特征依次是：

- impression video ID embedding：当前候选视频的 Embedding；
- watched video IDs average embedding：用户观看过的最后 N 个视频 Embedding 的平均值；
- language embedding：用户语言的 Embedding 和当前候选视频语言的 Embedding；
- time since last watch：表示用户上次观看同频道视频距今的时间；
- previous impressions：该视频已经被曝光给该用户的次数；

这 5 类特征连接起来之后，需要再经过三层 ReLU 网络进行充分的特征交叉，然后就到了输出层。这里重点注意，排序模型的输出层与候选集生成模型又有所不同。不同主要有两点：一是候选集生成模型选择了 softmax 作为其输出层，而排序模型选择了 weighted logistic regression（加权逻辑回归）作为模型输出层；二是候选集生成模型预测的是用户会点击“哪个视频”，排序模型预测的是用户“要不要点击当前视频”。

其实，排序模型采用不同输出层的根本原因就在于，YouTube 想要更精确地预测 用户的观看时长，因为观看时长才是 YouTube 最看中的商业指标，而使用 Weighted LR 作为输出层，就可以实现这样的目标。

在 Weighted LR 的训练中，我们需要为每个样本设置一个权重，权重的大小，代表了这个样本的重要程度。为了能够预估观看时长，YouTube 将正样本的权重设置为用户观看这个视频的时长，然后再用 Weighted LR 进行训练，就可以让模型学到用户观看时长的信息。

对于排序模型，必须使用 TensorFlow Serving 等模型服务平台，来进行模型的线上推断。

场景

对Youtube海量的视频进行快速、准确的排序。

优缺点

- 优点：结构较为简单，使用两级推荐结构从百万级别的视频候选集中进行视频推荐，且效果良好。
- 缺点：Youtube的深度推荐系统论文发布于2016年，现已成为各大公司的“基本操作”，按如今的标准已无新颖之处。

