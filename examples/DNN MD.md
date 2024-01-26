## 简介

CTR(Click Through Rate)，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。

简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。

CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。

本项目实现了使用 Deep Neural Network (DNN) 模型进行CTR预测

## 构建模型

CTR-DNN模型的组网比较直观，本质是一个二分类任务，代码参考`model.py`。模型主要组成是一个`Embedding`层，四个`FC`层，以及相应的分类任务的loss计算和auc计算。

### 数据输入
CTR-DNN模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_input_dim`指定，数据类型是归一化后的浮点型数据。`sparse_inputs`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`1~26`的26个稀疏参数输入，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

#### Embedding层
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`sparse_input`，由超参的`sparse_feature_number`和`sparse_feature_dimshape`定义。需要特别解释的是`is_sparse`参数，当我们指定`is_sprase=True`后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。

各个稀疏的输入通过Embedding层后，将其合并起来，置于一个list内，以方便进行concat的操作。

#### FC层
将离散数据通过embedding查表得到的值，与连续数据的输入进行`concat`操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了4层FC，每层FC的输出维度由超参`fc_sizes`指定，每层FC都后接一个`relu`激活函数，每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。

#### Loss及Auc计算
- 预测的结果通过一个输出shape为2的FC层给出，该FC层的激活函数是softmax，会给出每条样本分属于正负样本的概率。
- 每条样本的损失由交叉熵给出，交叉熵的输入维度为[batch_size,2]，数据类型为float，label的输入维度为[batch_size,1]，数据类型为int。
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc，auc的结果由`fluid.layers.auc()`给出，该层的返回值有三个，分别是从第一个batch累计到当前batch的全局auc: `auc`，最近几个batch的auc: `batch_auc`，以及auc_states: `_`，auc_states包含了`batch_stat_pos, batch_stat_neg, stat_pos, stat_neg`信息。`batch_auc`我们取近20个batch的平均，由参数`slide_steps=20`指定，roc曲线的离散化的临界数值设置为4096，由`num_thresholds=2**12`指定。

完成上述组网后，我们最终可以通过训练拿到`BATCH_AUC`与`auc`两个重要指标。
