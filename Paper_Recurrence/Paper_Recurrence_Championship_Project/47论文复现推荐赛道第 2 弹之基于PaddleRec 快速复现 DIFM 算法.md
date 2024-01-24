### 一、项目背景

   本项目是 [飞桨论文复现挑战赛（第四期）](https://aistudio.baidu.com/aistudio/competition/detail/106) 推荐赛道, 我所选做的第 2 篇. 第一篇, 可以参考 [论文复现赛: 基于PaddleRec 24小时快速复现经典 CTR 预估算法](https://aistudio.baidu.com/aistudio/projectdetail/2263714?contributionType=1&shared=1), 欢迎 fork, 欢迎提出批评建议~ 🎉🎉

本项目仍然是基于 PaddleRec 框架对论文进行复现, PaddleRec 框架相关介绍及使用方法可以参考上一篇相关章节. 下面👇 我们复现挑战赛的 94 号题目 [DIFM:A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf) 复现进行介绍. 主要分为以下几个部分:

- **DIFM 算法原理**
- **如何基于 PaddleRec 快速复现**
- **总结**
- **参考资料**
- **写在最后**


### 二、DIFM 算法原理

#### 1、简介

![DIFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffgzgk1bj30kq0e8wfz.jpg)

原论文：[A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf)

DIFM 是经典 CTR 预估算法 FM 的一个变种, 上图为 DIFM 的网络结构图. Paper 标题中所指的 Dual-FEN 为 `vector-wise` 和 `bit-wise`是两个 Input-aware Factorization 模块, 一个是 `bit-wise`, 一个是 `vector-wise`。只是维度上不同，实现的直觉是一样的。bit-wise 维度会对某一个 sparse embedding 向量内部彼此进行交叉，而 vector-wise 仅仅处理 embedding 向量层次交叉。

把这两个 FEN 模块拿掉, DIFM 就退化为 FM 算法了. 把 vector-wise FEN 模块去掉，DIFM 就退化为 IFM 模型，该算法也是论文作者实验组的大作，其结构图如下：

![IFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffi72287j60ez0cwq3p02.jpg)

两类不同维度的 FEN(Factor Estimating Net) 作用都是一致的，即输出 Embedding Layer 相应向量的权重。举个例子，假设上游有 n 个 sparse features， 则 FEN 输出结果为 [a1, a2, ..., an]. 在 Reweighting Layer 中，对原始输入进行权重调整。最后输入到 FM 层进行特征交叉，输出预测结果。

因此，总结两篇论文 IFM 和 DIFM 步骤如下：

- sparse features 经由 Embedding Layer 查表得到 embedding 向量，dense features 特征如何处理两篇论文都没提及；
- sparse features 对应的一阶权重也可以通过 1 维 Embedding Layer 查找；
- sparse embeddings 输入 FEN (bit-wise or vector-wise)，得到特征对应的权重 [a1, a2, ..., an]；
- Reweighting Layer 根据上一步骤中的特征权重，对 sparse embeddings 进一步调整；
- FM Layer 进行特征交叉，输出预测概率；


#### 2、复现精度

本项目实现了 IFM、 DIFM 以及在 IFM 基础上增加了 deep layer 用于处理 dense features, 记作 IFM-Plus 的三种模型.
在 DIFM 论文中，两种算法在 Criteo 数据集的表现如下：

![](https://tva1.sinaimg.cn/large/008i3skNly1gtfg698y4nj30bo06tdgp.jpg)

本次 PaddlePaddle 论文复现赛要求在 PaddleRec Criteo 数据集上，DIFM 的复现精度为 AUC > 0.799.

实际本项目复现精度为：
- IFM：AUC = 0.8016
- IFM-Plus: AUC = 0.8010
- DIFM: AUC = 0.799941


#### 3、数据集

原论文采用 Kaggle Criteo 数据集，为常用的 CTR 预估任务基准数据集。单条样本包括 13 列 dense features、 26 列 sparse features及 label.

[Kaggle Criteo 数据集](https://www.kaggle.com/c/criteo-display-ad-challenge)
- train set: 4584, 0617 条
- test set:   604, 2135 条 （no label)

[PaddleRec Criteo 数据集](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)
- train set: 4400, 0000 条
- test set:   184, 0617 条

P.S. 原论文所提及 Criteo 数据集为 Terabyte Criteo 数据集(即包含 1 亿条样本)，但作者并未使用全量数据，而是采样了连续 8 天数据进行训练和测试。
这个量级是和 PaddleRec Criteo 数据集是一样的，因此复现过程中直接选择了 PaddleRec 提供的数据。 原文表述如下：

![数据集介绍](https://tva1.sinaimg.cn/large/008i3skNly1gtgdgteholj61g40e6af502.jpg)


#### 4. 最优复现参数

```
  # 原文复现相关参数
  att_factor_dim: 80
  att_head_num: 16
  fen_layers_size:  [256, 256, 27]
  class: Adam
  learning_rate: 0.001
  train_batch_size: 2000
  epochs: 2

  # 简单调节 train_batch_size 到 1024，AUC 可以由 0.799941 提升到 0.801587
```

#### 5、复现记录
1. 参考 PaddleRec 中 FM， 实现 IFM 模型，全量 Criteo 测试集上 AUC = 0.8016；
2. 在 IFM 模型基础上，增加 dnn layer 处理 dense features, 全量 Criteo 测试集上 AUC = 0.8010；
3. 在 IFM 模型基础上，增加 Multi-Head Self Attention，实现 DIFM；0.799941；
4. 增加 Multi-Head Self Attention 模块后，会导致模型显著过拟合，需要进一步细致调参，本项目参数直接参考论文默认参数，并未进行细粒度参数调优；

### 三、如何基于 PaddleRec 快速复现

*P.S. 如果读者不了解 PaddleRec 动态图训练及推断过程, 强烈建议阅读上一篇项目的相关部分之后再继续.*

基于 PaddleRec 快速复现的关键是 net.py 模型组网. 这里介绍一下 net.py 代码:

#### 1. MLP

下面实现 MLP 层, 可以看到和 PyTorch、Tensorflow2 的语法非常接近, 几乎可以无缝切换到 PaddlePaddle. 官网 API 文档中有一张映射表, 可以参考, [PyTorch-PaddlePaddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html)

```Python
class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None, **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.l2 = l2
        self.mlp = []
        self.last_action = last_action

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                             initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
            else:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit)))
                self.mlp.append(dense)

                if last_action is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs
```

#### 2. Multi-Head Attention Layer (Vector-wise FEN)

![vector-wise part](https://tva1.sinaimg.cn/large/008i3skNly1guai5unvhwj60fj0dv3zt02.jpg)

读过论文的同学知道, 上述的 vector-wise 是通过一个 Multi-Head Attention 模块和 Res-Net 残差网络实现的. 具体实现可以直接看下面👇代码, 该模块的输出是 input embedding 向量的权重.

```python
class MultiHeadAttentionLayer(nn.Layer):
    def __init__(self,
                 att_factor_dim,
                 att_head_num,
                 sparse_feature_dim,
                 sparse_field_num,
                 batch_size):
        super(MultiHeadAttentionLayer, self).__init__()
        self.att_factor_dim = att_factor_dim
        self.att_head_num = att_head_num
        self.sparse_feature_dim = sparse_feature_dim
        self.sparse_field_num = sparse_field_num
        self.batch_size = batch_size

        self.W_Query = paddle.create_parameter(default_initializer=nn.initializer.TruncatedNormal(),
                                               shape=[self.sparse_feature_dim, self.att_factor_dim * self.att_head_num],
                                               dtype='float32')
        self.W_Key = paddle.create_parameter(default_initializer=nn.initializer.TruncatedNormal(),
                                             shape=[self.sparse_feature_dim, self.att_factor_dim * self.att_head_num],
                                             dtype='float32')
        self.W_Value = paddle.create_parameter(default_initializer=nn.initializer.TruncatedNormal(),
                                               shape=[self.sparse_feature_dim, self.att_factor_dim * self.att_head_num],
                                               dtype='float32')
        self.W_Res = paddle.create_parameter(default_initializer=nn.initializer.TruncatedNormal(),
                                             shape=[self.sparse_feature_dim, self.att_factor_dim * self.att_head_num],
                                             dtype='float32')
        self.dnn_layer = MLPLayer(input_shape=(self.sparse_field_num + 1) * self.att_factor_dim * self.att_head_num,
                                  units_list=[self.sparse_field_num + 1],
                                  last_action="relu")

    def forward(self, combined_features):
        """
        combined_features: (batch_size, (sparse_field_num + 1), embedding_size)
        W_Query: (embedding_size, factor_dim * att_head_num)
        (b, f, e) * (e, d*h) -> (b, f, d*h)
        """
        # (b, f, d*h)
        querys = paddle.matmul(combined_features, self.W_Query)
        keys = paddle.matmul(combined_features, self.W_Key)
        values = paddle.matmul(combined_features, self.W_Value)

        # (h, b, f, d) <- (b, f, d)
        querys = paddle.stack(paddle.split(querys, self.att_head_num, axis=2))
        keys = paddle.stack(paddle.split(keys, self.att_head_num, axis=2))
        values = paddle.stack(paddle.split(values, self.att_head_num, axis=2))

        # (h, b, f, f)
        inner_product = paddle.matmul(querys, keys, transpose_y=True)
        inner_product /= self.att_factor_dim ** 0.5
        normalized_att_scores = Fun.softmax(inner_product)

        # (h, b, f, d)
        result = paddle.matmul(normalized_att_scores, values)
        result = paddle.concat(paddle.split(result, self.att_head_num, axis=0), axis=-1)

        # (b, f, h * d)
        result = paddle.squeeze(result, axis=0)
        result += paddle.matmul(combined_features, self.W_Res)

        # (b, f * h * d)
        result = paddle.reshape(result, shape=(self.batch_size, -1))
        m_vec = self.dnn_layer(result)
        return m_vec
```

#### 3. Bit-wise FEN

Bit-wise FEN 的实现相对简单一些, 直接拼接 input embedding 输入到 MLP 中进行隐式交叉, 最后一层维度设置为输入 embedding 的个数. 具体直接查看 `FENLayer` 中的相关部分:

```python
# -------------------- fen layer ------------------------------------
    # (batch_size, embedding_size)
    dense_embedding = self.dense_mlp(dense_inputs)
    dnn_logits = self.dnn_mlp(dense_embedding)
    dense_embedding = paddle.unsqueeze(dense_embedding, axis=1)

    # (batch_size, sparse_field_num, embedding_size)
    sparse_embedding = self.sparse_embedding(sparse_inputs_concat)

    # (batch_size, (sparse_field_num + 1), embedding_size)
    feat_embeddings = paddle.concat([dense_embedding, sparse_embedding], axis=1)

    # (batch_size, (sparse_field_num + 1))
    # m_x 这里输出的是相应权重向量
    m_x = self.fen_mlp(paddle.reshape(feat_embeddings, shape=(self.batch_size, -1)))
```

#### 4. IFM

该部分介绍一下, 如何组网构建 IFM 模型. input embedding 经由上一部分 Bit-wise FEN 模块得到相应的权重向量, 再对输入 embedding 向量进行调整. 接着输入到 FM 模块中进行特征交叉, 最后输出预测值.

```
class IFM(nn.Layer):
    def __init__(self,
                 sparse_field_num,
                 sparse_feature_num,
                 sparse_feature_dim,
                 dense_feature_dim,
                 fen_layers_size,
                 dense_layers_size,
                 batch_size):
        super(IFM, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size
        self.dense_layers_size = dense_layers_size
        self.batch_size = batch_size
        self.fen_layer = FENLayer(sparse_field_num=self.sparse_field_num,
                                  sparse_feature_num=self.sparse_feature_num,
                                  sparse_feature_dim=self.sparse_feature_dim,
                                  dense_feature_dim=self.dense_feature_dim,
                                  fen_layers_size=self.fen_layers_size,
                                  dense_layers_size=self.dense_layers_size,
                                  batch_size=self.batch_size)
        self.fm_layer = FMLayer()

    def forward(self, sparse_inputs, dense_inputs):
        dnn_logits, feat_emb_one, feat_embeddings, m_x = self.fen_layer(sparse_inputs, dense_inputs)

        m_x = Fun.softmax(m_x)

        # (batch_size, (sparse_field_num + 1))
        feat_emb_one = feat_emb_one * m_x
        # (batch_size, (sparse_field_num + 1), embedding_size)
        feat_embeddings = feat_embeddings * paddle.unsqueeze(m_x, axis=-1)

        # (batch_size, 1)
        first_order = paddle.sum(feat_emb_one, axis=1, keepdim=True)

        return self.fm_layer(dnn_logits, first_order, feat_embeddings)
```

#### 5. DIFM
该部分介绍 DIFM 的组网方式, 与 IFM 类似, 只是 forward 中输入向量权重变成了 vector-wise 和 bit-wise 的和, 即 `m = Fun.softmax(m_vec + m_bit)`. 具体源码参考 DIFM 中 forward 部分, 如下:

```python
    def forward(self, sparse_inputs, dense_inputs):
        dnn_logits, feat_emb_one, feat_embeddings, m_bit = self.fen_layer(sparse_inputs, dense_inputs)
        m_vec = self.mha_layer(feat_embeddings)
        # 融合 vector-wise 和 bit-wise 两部分 FEN 权重
        m = Fun.softmax(m_vec + m_bit)

        feat_emb_one = feat_emb_one * m
        feat_embeddings = feat_embeddings * paddle.unsqueeze(m, axis=-1)

        first_order = paddle.sum(feat_emb_one, axis=1, keepdim=True)

        return self.fm_layer(dnn_logits, first_order, feat_embeddings)
```

其他模块, 如 loss 、metrics 等计算集成在 PaddleRec 框架中, 可以直接查看源码.

#### 6. 环境依赖

- 硬件：CPU、GPU
- 框架：
  - PaddlePaddle >= 2.1.2
  - Python >= 3.7

#### 7、代码结构与详细说明

代码结构遵循 PaddleRec 框架结构
```
|--models
  |--rank
    |--difm                   # 本项目核心代码
      |--data                 # 采样小数据集
      |--config.yaml          # 采样小数据集模型配置
      |--config_bigdata.yaml  # Kaggle Criteo 全量数据集模型配置
      |--criteo_reader.py     # dataset加载类  
      |--dygraph_model.py     # PaddleRec 动态图模型训练类
      |--net.py               # difm 核心算法代码，包括 difm 组网、ifm 组网等
|--tools                      # PaddleRec 工具类
|--LICENSE                    # 项目 LICENSE
|--README.md                  # readme
|--run.sh                     # 项目执行脚本(需在 aistudio notebook 中运行)
```

#### 8. 快速开始

也可以在 AI-Studio 的 NoteBook 上 clone 代码, 直接上手跑跑看, 步骤如下:

- Step 1, git clone code
- Step 2, download data
- Step 3, train model & infer

```
################# Step 1, git clone code ################
# 当前处于 /home/aistudio 目录, 代码存放在 /home/work/rank/DIFM-Paddle 中

import os
if not os.path.isdir('work/rank/DIFM-Paddle'):
    if not os.path.isdir('work/rank'):
        !mkdir work/rank
    !cd work/rank && git clone https://hub.fastgit.org/Andy1314Chen/DIFM-Paddle.git

################# Step 2, download data ################
# 当前处于 /home/aistudio 目录，数据存放在 /home/data/criteo 中

import os
os.makedirs('data/criteo', exist_ok=True)

# Download  data
if not os.path.exists('data/criteo/slot_test_data_full.tar.gz') or not os.path.exists('data/criteo/slot_train_data_full.tar.gz'):
    !cd data/criteo && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
    !cd data/criteo && tar xzvf slot_test_data_full.tar.gz

    !cd data/criteo && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz
    !cd data/criteo && tar xzvf slot_train_data_full.tar.gz

################## Step 3, train model ##################
# 启动训练脚本 (需注意当前是否是 GPU 环境）
!cd work/rank/DIFM-Paddle && sh run.sh config_bigdata

```

#### 2. criteo slot_test_data_full 验证集结果
```
...
2021-08-14 11:53:10,026 - INFO - epoch: 0 done, auc: 0.799622, epoch time: 261.84 s
2021-08-14 11:57:32,841 - INFO - epoch: 1 done, auc: 0.799941, epoch time: 262.81 s
```

#### 3. 使用预训练模型进行预测
- 在 notebook 中切换到 `V1.2调参数` 版本，加载预训练模型文件，可快速验证测试集 AUC；
- ！！注意 config_bigdata.yaml 的 `use_gpu` 配置需要与当前运行环境保存一致
```
!cd /home/aistudio/work/rank/DIFM-Paddle && python -u tools/infer.py -m models/rank/difm/config_bigdata.yaml
```




### 四、总结

直接看 DIFM 论文, 发现该模型其实是比较复杂的, 又是 Multi-Head Attention 又是 Res-Net 等, 😵‍💫眼花缭乱的. 但只要仔细拆解, 会发现论文作者实验室之前有篇类似工作, 可以看作是该篇论文的前传, IFM.

将 FM、IFM、DIFM 三篇论文网络结构放在一起, 你会发现可以想垒积木一样, 一点点进行模型组网. 比如, FM 加上 bit-wise FEN(其实是 MLP 模块) 构成了 IFM, 在 IFM 基础上, 加上 vector-wise FEN(其实是 Multi-Head Attention 模块)构成了 DIFM. 而 PaddleRec 框架中已经集成了很多类似 FM 一样的基础经典 CTR 模型.


所以, 复现论文和阅读论文也有一点共通之处, 即善于挖掘论文的创新点(相对 baseline, 改进之处在哪), 在前人的基础上快速复现.

### 五、参考资料

1. DIFM [A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf)
2. IFM [An Input-aware Factorization Machine for Sparse Prediction](extension://oikmahiipjniocckomdccmplodldodja/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.ijcai.org%2Fproceedings%2F2019%2F0203.pdf#=&zoom=100)
3. DIFM 的 Tensorflow 实现 [https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/difm.py](https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/difm.py)
4. IFM 的 Tensorflow 实现 [https://github.com/gulyfish/Input-aware-Factorization-Machine/blob/master/code/IFM.py](https://github.com/gulyfish/Input-aware-Factorization-Machine/blob/master/code/IFM.py)
5. PaddleRec 框架 [https://github.com/PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
6. [飞桨论文复现打卡营](https://aistudio.baidu.com/aistudio/education/group/info/24681)

### 六、写在最后

最后... 如果各位大佬感觉上述内容有点儿帮助, 劳烦去 [github](https://github.com/Andy1314Chen/DIFM-Paddle) 给个 star, 爱你呦~ ♥️


```python

```
