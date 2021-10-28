### 一、项目背景

   偶然看到了[飞桨论文复现挑战赛（第四期）](https://aistudio.baidu.com/aistudio/competition/detail/106), 抱着划水的态度, 报名参加了一个推荐赛题. 因为本身已经参加工作了, 没有太多空闲时间, 只能晚上下班或者周末打个酱油, 划划水~😂

在实际推荐算法开发工作中, 一般也都有自己的开发项目框架, 包含了`数据加载`、`特征处理`、`模型构建`等模块, 可以快速完成一个新算法的开, 类似 GitHub 上开源的 DeepCTR. 因此, 首先找了一下 PaddlePaddle 的相关项目, 所幸 PaddlePaddle 团队有个 [PaddleRec](https://github.com/PaddlePaddle/PaddleRec).

如果已经对论文比较熟悉了, 那基于 PaddleRec 开发就十分方便了, 甚至都不用题目里提到的 24 小时. 下面👇 我们根据复现挑战赛的 93 号题目 DLRM 复现进行介绍. 主要分为以下几个部分:

- **DLRM 算法原理**
- **PaddleRec 框架**
- **如何基于 PaddleRec 快速复现**
- **总结**
- **参考资料**


### 二、DLRM 算法原理

![DLRM](https://tva1.sinaimg.cn/large/008i3skNly1gt8kwo40g9j30ei0cmjru.jpg)

1. 模型结构

推荐 rank 模型一般较为简单，如上图 DLRM 的网络结构看着和 DNN 就没啥区别，主要由四个基础模块构成，`Embeddings`、 `Matrix Factorization`、`Factorization Machine`和`Multilayer Perceptrons`。

DLRM 模型的特征输入，主要包括 dense 数值型和 sparse 类别型两种特征。dense features 直接连接 MLP（如图中的蓝色三角形），
sparse features 经由 embedding 层查找得到相应的 embedding 向量。Interactions 层进行特征交叉（包含 dense features 和 sparse features 的交叉及
sparse features之间的交叉等），与因子分解机 FM 有些类似。

DLRM 模型中所有的 sparse features 的 embedding 向量长度均是相等的，且dense features 经由 MLP 也转化成相同的维度。这点是理解该模型代码的关键。

- dense features 经过 MLP (bottom-MLP) 处理为同样维度的向量
- spare features 经由 lookup 获得统一维度的 embedding 向量（可选择每一特征对应的 embedding 是否经过 MLP 处理）
- dense features & sparse features 的向量两两之间进行 dot product 交叉
- 交叉结果再和 dense 向量 concat 一起输入到顶层 MLP (top-MLP)  
- 经过 sigmoid 函数激活得到点击概率

2. Experiments

大佬发文章就是 NB，DLRM vs DCN without extensive tuning and no regularization is used. 简简单单的 SGD + lr=0.1
就把 Accuracy 干上去了。。。

![实验结果](https://tva1.sinaimg.cn/large/008i3skNly1gta7vj34mkj30ty0c8abt.jpg)

3. 原论文 repo

[https://github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm)

### 三、PaddleRec 框架介绍

PaddleRec 涵盖了推荐系统的各个阶段, 包括内容理解、匹配、召回、排序、多任务、重排序等, 但这里我们只关注 CTR 预估, 即排序阶段.该部分在 models/rank/ 路径下, 已经实现了 `deepfm`、`dnn`、`ffm`、`fm`等经典 CTR 算法, 每类算法包含静态图和动态图两种训练方式. 我们一般选择动态图复现, 因为和 PyTorch 及 Tensorflow2 等语法上更接近, 调试也更方便.

我们在 models/rank/ 路径下定义 dataset 加载和 模型组网方式之后, 便可以通过 PaddleRec 下 tools 类进行模型的训练及预测. 一个简单的 DNN 算法训练和推断就是下面简单的两行命令:

```python
# Step 1, 训练模型
python -u tools/trainer.py -m models/rank/dnn/config.yaml

# Step 2, 预测推断
python -u tools/infer.py -m models/rank/dnn/config.yaml
```

以上 trainer.py 和 infer.py 都是 PaddleRec 框架预先实现的训练类和预测类, 我们不需要关心细节, 只需关注数据加载及模型组网等就行, 通过上述的配置文件 config.yaml 去调用我们实现的数据读取类和模型.


```
|--models
  |--rank
    |--dlrm                   # 本项目核心代码
      |--data                 # 采样小数据集
      |--config.yaml          # 采样小数据集模型配置
      |--config_bigdata.yaml  # Kaggle Criteo 全量数据集模型配置
      |--criteo_reader.py     # dataset加载类  
      |--dygraph_model.py     # PaddleRec 动态图模型训练类
      |--net.py               # dlrm 核心算法代码，包括 dlrm 组网等
|--tools                      # PaddleRec 工具类
```

总结一下, 基于 PaddleRec CTR 模型快速复现只需要我们在 models/rank/ 路径下, 新建自己的模型文件夹, 比如我这里的 dlrm/. 其中, 最重要的三个是
- config.yaml 数据、特征、模型等配置
- xxxx_reader.py 数据集加载方式
- net.py 模型组网

因为 DLRM 复现要求的是 Criteo 数据集, 甚至这个 reader 都不用自己去写, PaddleRec 帮你做好了. 更多关于 PaddleRec 的介绍, 可以参考这里 [https://github.com/PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec).

### 四、如何基于 PaddleRec 快速复现
上文提到, 基于 PaddleRec 快速复现的关键是 net.py 模型组网. 这里介绍一下 net.py 代码:

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


下面是 DLRM 模型的核心组网, 代码中有注释, 结合第二部分算法原理很容易理解.

在 **\_\_init\_\_** 初始化函数中, 定义 bottom-MLP 模块处理数值型特征, 定义 Embedding 层完成稀疏特征到 Embedding 向量的映射. 定义 top-MLP 模块处理交叉特征的进一步泛化, 得到 CTR 预测值.

在 **forward** 中, 对输入的 dense features 和 sparse features 进行处理, 分别得到的 embedding 向量拼接在一起. 经过 vector-wise 特征交叉后, 输入 top-MLP 得到预测值.

```Python
class DLRMLayer(nn.Layer):
    def __init__(self,
                 dense_feature_dim,
                 bot_layer_sizes,
                 sparse_feature_number,
                 sparse_feature_dim,
                 top_layer_sizes,
                 num_field,
                 sync_mode=None):
        super(DLRMLayer, self).__init__()
        self.dense_feature_dim = dense_feature_dim
        self.bot_layer_sizes = bot_layer_sizes
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.top_layer_sizes = top_layer_sizes
        self.num_field = num_field

        # 定义 DLRM 模型的 Bot-MLP 层
        self.bot_mlp = MLPLayer(input_shape=dense_feature_dim,
                                units_list=bot_layer_sizes,
                                last_action="relu")

		# 定义 DLRM 模型的 Top-MLP 层
        self.top_mlp = MLPLayer(input_shape=int(num_field * (num_field + 1) / 2) + sparse_feature_dim,
                                units_list=top_layer_sizes)

		# 定义 DLRM 模型的 Embedding 层
        self.embedding = paddle.nn.Embedding(num_embeddings=self.sparse_feature_number,
                                             embedding_dim=self.sparse_feature_dim,
                                             sparse=True,
                                             weight_attr=paddle.ParamAttr(
                                                 name="SparseFeatFactors",
                                                 initializer=paddle.nn.initializer.Uniform()))

    def forward(self, sparse_inputs, dense_inputs):
        # (batch_size, sparse_feature_dim)
        x = self.bot_mlp(dense_inputs)

        # interact dense and sparse feature
        batch_size, d = x.shape

        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)
        # 拼接数值型特征和 Embedding 特征
        T = paddle.reshape(paddle.concat(x=sparse_embs + [x], axis=1), (batch_size, -1, d))
        # 进行 vector-wise 特征交叉
        Z = paddle.bmm(T, paddle.transpose(T, perm=[0, 2, 1]))

        Zflat = paddle.triu(Z, 1) + paddle.tril(paddle.ones_like(Z) * MIN_FLOAT, 0)
        Zflat = paddle.reshape(paddle.masked_select(Zflat,
                                                    paddle.greater_than(Zflat, paddle.ones_like(Zflat) * MIN_FLOAT)),
                               (batch_size, -1))

        R = paddle.concat([x] + [Zflat], axis=1)
		# 交叉特征输入 Top-MLP 进行 CTR 预测
        y = self.top_mlp(R)
        return y
```

可以在 AI-Studio 的 NoteBook 上 clone 代码, 直接上手跑跑看, 步骤如下:

- Step 1, git clone code
- Step 2, download data
- Step 3, train model & infer

```
################# Step 1, git clone code ################
# 当前处于 /home/aistudio 目录, 代码存放在 /home/work/rank/DLRM-Paddle 中

import os
if not os.path.isdir('work/rank/DLRM-Paddle'):
    if not os.path.isdir('work/rank'):
        !mkdir work/rank
    # 国内访问或 git clone 较慢, 利用 hub.fastgit.org 加速
    !cd work/rank && git clone https://hub.fastgit.org/Andy1314Chen/DLRM-Paddle.git
```

```
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
```

```
################## Step 3, train model ##################
# 启动训练脚本 (需注意当前是否是 GPU 环境, 非 GPU 环境请修改 config_bigdata.yaml 配置中 use_gpu 为 False）
!cd work/rank/DLRM-Paddle && sh run.sh config_bigdata
```

### 五、总结
1. 基于 PaddleRec 框架可以快速进行推荐算法的复现, 让你更加专注模型的细节, 提升复现效率;
2. PaddleRec 封装了训练及推断过程, 提升了开发的速度, 但是也隐藏了一些细节, 如何提高数据加载速度? 如何在训练过程中设置 easy_stopping? 等等问题还需要仔细阅读 PaddleRec 源码去了解.


### 六、参考资料
1. 原论文 [Deep Learning Recommendation Model for Personalization and Recommendation Systems](extension://oikmahiipjniocckomdccmplodldodja/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1906.00091v1.pdf)
2. PyTorch 实现 [https://github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm)
3. PaddleRec 框架 [https://github.com/PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
4. [飞桨论文复现打卡营](https://aistudio.baidu.com/aistudio/education/group/info/24681)

### 七、写在最后
最后... 如果各位大佬感觉上述内容有点儿帮助, 麻烦 [github](https://github.com/Andy1314Chen/DLRM-Paddle) 给个 star, 爱你呦~ ♥️

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
