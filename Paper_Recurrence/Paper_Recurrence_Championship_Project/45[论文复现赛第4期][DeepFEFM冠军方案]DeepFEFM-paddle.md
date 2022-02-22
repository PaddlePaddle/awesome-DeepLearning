# [论文复现赛第4期][DeepFEFM冠军方案]DeepFEFM-paddle

## 项目简介
本项目为基于PaddlePaddle 2.1.0 复现 DeepFeFm 论文，基于 [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 二次开发。

论文地址：[Field-Embedded Factorization Machines for Click-through rate prediction](https://arxiv.org/pdf/2009.09931v2.pdf)

原论文代码：[DeepCTR-DeepFEFM](https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/deepfefm.py)

Paddle版本代码：[PaddleRec版本，含静态图和动态图代码](https://github.com/thinkall/PaddleRec/tree/master/models/rank/deepfefm), [不依赖PaddleRec版本，只含动态图代码](https://github.com/thinkall/deepfefm)

预训练模型下载：链接: https://pan.baidu.com/s/1CftnEt0nl1V6w6ApDzqkKA 提取码: dyvf


## 模型简介

`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了deepFEFM模型。

该模型是FM类模型的又一变种。模型架构如下：

<div align="center"><img src=https://ai-studio-static-online.cdn.bcebos.com/0af42f47b9a14c1f9e8482459561d1f200b9e43933484c7093ed1866d84b5e9b height=500></img></div>

模型的核心FEFM公式如下：

<div align="center"><img src=https://ai-studio-static-online.cdn.bcebos.com/338dca71323a4816981f2218a8640b26e6621cd18b3c47c1bbf14fa30530a3ec height=70></img></div>

其中与FFM，FwFM模型的核心区别就是使用一个对称矩阵 Field pair matrix embeddings `$W_{F(i),F(j)}$` 对不同field的关系进行建模。

### 一阶项部分
一阶项部分类似于我们rank下的logistic_regression模型。主要由embedding层和reduce_sum层组成。
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`feat_idx`，shape由超参的`sparse_feature_number`定义。  
各个稀疏的输入通过Embedding层后，进行reshape操作，方便和连续值进行结合。  
将离散数据通过embedding查表得到的值，与连续数据的输入进行相乘再累加的操作，合为一个一阶项的整体。  
用公式表示如下：  

<div align="center"><img src=https://ai-studio-static-online.cdn.bcebos.com/647fa7ecb7c1463293b68f1d63c23dc34e486b68a80749fa8eb0bf6fce670b9f height=30></img></div>

### 二阶项部分
二阶项部分主要实现了公式中的交叉项部分，也就是特征的组合部分。

<div align="center"><img src=https://ai-studio-static-online.cdn.bcebos.com/bf57f58bf79f454fa8f1d4d91c167b80ced880afe2fa4f89a270731c93bd383f height=30></img></div>

### DNN部分
相比fm模型，我们去除了fm模型中的偏移量，而加入了dnn部分作为特征间的高阶组合，通过并行的方式组合fm和dnn两种方法，两者共用底层的embedding数据。dnn部分的主要组成为三个全连接层，每层FC的输出维度都为1024，每层FC都后接一个relu激活函数，每层FC的初始化方式为符合正态分布的随机初始化，每层FC都加入L2正则化。  
最后接了一层输出维度为1的fc层，方便与fm部分综合计算预测值。  

### Loss及Auc计算
- 预测的结果将FM的一阶项部分，二阶项部分以及dnn部分相加，再通过激活函数sigmoid给出，为了得到每条样本分属于正负样本的概率，我们将预测结果和`1-predict`合并起来得到predict_2d，以便接下来计算auc。  
- 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。  
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc指标。

## 复现精度

| 数据集 | 复现精度 |
| --- | --- |
| [Criteo(Paddle版)](https://github.com/PaddlePaddle/PaddleRec/blob/master/datasets/criteo/run.sh) | 0.80276 |

- 核心参数设置
```
- lr: 0.0005
- batch_size: 5120
- optimizer: Adam
```

## 复现步骤
### 下载 PaddleRec


```python
import os

current_path = os.path.realpath('.')
print(current_path)

if not os.path.exists('PaddleRec'):
    # !git clone https://github.com/PaddlePaddle/PaddleRec.git
    !unzip PaddleRec.zip -d PaddleRec
else:
    print('PaddleRec already exist!')
    !cd PaddleRec/ && git checkout master
```

### 准备DeepFEFM模型文件
这里是基于PaddleRec中已有的deepfm模型进行修改


```python
%cd PaddleRec/models/rank/
if not os.path.exists('deepfefm'):
    !cp -r deepfm deepfefm
%cd deepfefm
# !rm -r picture/
```

### 实现DeepFEFM模型代码

#### 修改 net.py
这里定义模型的核心组网代码：

模型由两部分组成，fefm部分和dnn部分，其中fefm部分又分为一阶项和二阶项部分。最终的预测是一阶项，二阶项和dnn输出之和再过sigmoid激活函数得到。
```
    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order, dnn_input = self.fefm(sparse_inputs,
                                                                 dense_inputs)

        y_dnn = self.dnn(dnn_input)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict
```

不同于deepfm，这里需要对dense feature也做离散化，即在特征交叉时每个dense feature也会与sparse feature一样参与。所以这里sparse_feature_number其实是sparse feature的数量与离散化后的dense feature的数量之和。
```
        # sparse embedding and dense embedding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=False,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))),
                regularizer=paddle.regularizer.L2Decay(1e-6)))
```

这里做二阶项时，dense_inputs_re 就是在对denser feature做离散化，离散化为 1000002 ~ 1100002 之间的整数。所以对于Criteo数据集，这里最后是39个特征进行交叉，二不是deepfm中的26个。

**需要注意的是，原文中的离散化不是这么做的，但由于我为了直接使用paddleRec提供的数据集，所以是基于paddleRec处理完的数据集再将已经归一化的dense feature离散化。如果完全使用原论文一致的处理方式，模型精度应该会更高些。**
```
        # -------------------- Field-embedded second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)  # [batch_size, sparse_feature_number, sparse_feature_dim]
        dense_inputs_re = (dense_inputs * 1e5 + 1e6 + 2).astype('int64')  # [batch_size, dense_feature_number]
        dense_embeddings = self.embedding(dense_inputs_re)  # [batch_size, dense_feature_number, dense_feature_dim]

        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings], 1)  # [batch_size, dense_feature_number + sparse_feature_number, dense_feature_dim]
```

`$W_{F(i),F(j)}$`需要是对称矩阵，这里通过将其分解成了一个矩阵 `field_pair_embed_ij` 与其自身的转置之和，这样就确保了对称性。
```
        for fi, fj in itertools.combinations(range(self.num_fields), 2):  # self.num_fields = 39, dense_feature_number + sparse_num_field
            field_pair_id = str(fi) + "-" + str(fj)
            feat_embed_i = paddle.squeeze(feat_embeddings[0:, fi:fi + 1, 0:], axis=1)  # feat_embeddings: [batch_size, num_fields, sparse_feature_dim]
            feat_embed_j = paddle.squeeze(feat_embeddings[0:, fj:fj + 1, 0:], axis=1)  # [batch_size * sparse_feature_dim]
            field_pair_embed_ij = self.field_embeddings[field_pair_id]  # self.field_embeddings [sparse_feature_dim, sparse_feature_dim]

            feat_embed_i_tr = paddle.matmul(feat_embed_i, field_pair_embed_ij + paddle.transpose(field_pair_embed_ij, [1, 0]))  # [batch_size * embedding_size]

            f = batch_dot(feat_embed_i_tr, feat_embed_j, axes=1)  # [batch_size * 1]
            pairwise_inner_prods.append(f)
```

#### 缺失算子 batch_dot
batch_dot是tensorflow自带算子，这里我们基于paddle基础算子将其组合出来。代码在 myutils.py 里。

```
def batch_dot(x, y, axes=None):
    """Batchwise dot product.
    >>> x_batch = paddle.ones(shape=(32, 20, 1))
    >>> y_batch = paddle.ones(shape=(32, 30, 20))
    >>> xy_batch_dot = batch_dot(x_batch, y_batch, axes=(1, 2))
    >>> xy_batch_dot.shape
    (32, 1, 30)

    Shape inference:
    Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
    If `axes` is (1, 2), to find the output shape of resultant tensor,
        loop through each dimension in `x`'s shape and `y`'s shape:
    * `x.shape[0]` : 100 : append to output shape
    * `x.shape[1]` : 20 : do not append to output shape,
        dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
    * `y.shape[0]` : 100 : do not append to output shape,
        always ignore first dimension of `y`
    * `y.shape[1]` : 30 : append to output shape
    * `y.shape[2]` : 20 : do not append to output shape,
        dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
    `output_shape` = `(100, 30)`
    """
```

#### 修改 dygraph_model.py
修改model初始化为deepfefm：
```
    # define model
    def create_model(self, config):
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        fc_sizes = config.get("hyper_parameters.fc_sizes")
        sparse_fea_num = config.get('hyper_parameters.sparse_fea_num')
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')

        deepfefm_model = net.DeepFEFMLayer(sparse_feature_number,
                                       sparse_feature_dim, dense_feature_dim,
                                       sparse_input_slot - 1, fc_sizes)

        return deepfefm_model
```

增加对更多优化器的支持：
```
    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        opt = config.get("hyper_parameters.optimizer.class", 'Adam')
        if opt == 'Adagrad':
            optimizer = paddle.optimizer.Adagrad(
                learning_rate=lr, parameters=dy_model.parameters())
        elif opt == 'AdamW':
            optimizer = paddle.optimizer.AdamW(
                learning_rate=lr, parameters=dy_model.parameters())
        elif opt == 'Adam':
            optimizer = paddle.optimizer.Adam(
                learning_rate=lr, parameters=dy_model.parameters())
        else:
            raise NotImplementedError
        return optimizer
```

#### 修改trainer.py

加入 infer_test 可以边训练边用验证集验证，更好地保存最优结果：
```
def infer_test(dy_model, test_dataloader, dy_model_class, config,
               print_interval, epoch_id):
    metric_list, metric_list_name = dy_model_class.create_metrics()
    paddle.seed(12345)
    dy_model.eval()
    interval_begin = time.time()
    for batch_id, batch in enumerate(test_dataloader()):
        batch_size = len(batch[0])

        metric_list, tensor_print_dict = dy_model_class.infer_forward(
            dy_model, metric_list, batch, config)

        # only test 10 epochs to improve speed
        if batch_id > 10:
            break

    metric_str = ""
    for metric_id in range(len(metric_list_name)):
        metric_str += (metric_list_name[metric_id] +
                       ": {:.6f},".format(metric_list[metric_id].accumulate()))

    tensor_print_str = ""
    if tensor_print_dict is not None:
        for var_name, var in tensor_print_dict.items():
            tensor_print_str += (
                "{}:".format(var_name) + str(var.numpy()) + ",")

    logger.info("validation epoch: {} done, ".format(epoch_id) + metric_str +
                tensor_print_str + " epoch time: {:.2f} s".format(time.time(
                ) - interval_begin))

    dy_model.train()
    return metric_list[0].accumulate()
```

增加学习率遍历功能，方便找最优学习率：
```
    for lr in try_lrs:
        best_auc, best_lr = f(best_auc, best_lr, lr, args)
        reset_graph()
        if best_auc >= 0.8001:  # 0.81405 is the metric in the original paper
            break
```

#### 修改配置文件
通过`config/*.yaml`文件设置训练和评估相关参数，具体参数如下：
|  参数   | 默认值  | 说明 |
|  ----  |  ----  |  ----  |
|runner.train_data_dir|"data/sample_data/train"|训练数据所在文件夹|
|runer.train_reader_path|"criteo_reader"|训练数据集载入代码|
|runer.use_gpu|True|是否使用GPU|
|runer.train_batch_size|5120|训练时batch_size|
|runer.epochs|1|训练几个epoch|
|runner.print_interval|50|多少个batch打印一次信息|
|runner.model_init_path|"output_model_dmr/0"|继续训练时模型载入目录，默认未启用|
|runner.model_save_path|"output_model_dmr"|模型保存目录|
|runner.test_data_dir|"data/sample_data/test"|测试数据文件夹|
|runner.infer_reader_path| "alimama_reader"|测试数据集载入代码|
|runner.infer_batch_size|256|评估推理时batch_size|
|runner.infer_load_path|"output_model_dmr"|评估推理时模型载入目录|
|runner.infer_start_epoch|1000|评估推理的从哪个epoch开始，默认会把最优模型保存到目录1000，所以默认从1000开始，当然也可以从0开始|
|runner.infer_end_epoch|1001|评估推理到哪个epoch（不含）停止，默认值实际上只评估1000目录中的这1个模型|
|hyper_parameters.optimizer.class|Adam|优化器，Adam效果最好|
|hyper_parameters.optimizer.learning_rate|0.0005|学习率，应与batchsize同步调整|
|hyper_parameters.sparse_feature_dim|48|Embedding长度|
|hyper_parameters.fc_sizes|[1024, 1024, 1024]|隐藏层大小|

### 保存代码文件
这里将修改好的代码都放到了 work 目录下。然后将其复制到paddleRec文件夹下对应目录。后续便可以开始运行代码了。


```python
!cp $current_path/work/* $current_path/PaddleRec/models/rank/deepfefm/
!cp $current_path/work/trainer.py $current_path/PaddleRec/tools/
%cd $current_path/PaddleRec/models/rank/deepfefm
```

## 模型训练和测试

### 小样本训练和测试


```python
# 动态图训练
!python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml
```


```python
# 动态图预测
# !python -u ../../../tools/infer.py -m config.yaml
```

### 备份代码
如果直接修改了paddleRec中的代码，可以运行以下命令备份代码和配置文件。


```python
# # backup config
# !cp config*.yaml ~/work/
# !cp net.py ~/work/
# !cp dygraph_model.py ~/work/
# !cp myutils.py ~/work/
# !cp ~/PaddleRec/tools/trainer.py ~/work/
```

### 准备全量数据


```python
if not os.path.exists('/home/aistudio/data/data103052/slot_train_data_full/part-0'):
    !tar xzf /home/aistudio/data/data103052/slot_test_data_full.tar.gz -C /home/aistudio/data
    !tar xzf /home/aistudio/data/data103052/slot_train_data_full.tar.gz -C /home/aistudio/data
else:
    print('dataset is ready!')
```

### 全量数据训练和测试

- 部分训练日志：
```
2021-08-13 15:52:32,885 - INFO - **************common.configs**********
2021-08-13 15:52:32,885 - INFO - use_gpu: True, use_visual: False, train_batch_size: 5120, train_data_dir: /home/aistudio/data/slot_train_data_full, epochs: 4, print_interval: 100, model_save_path: output_model_all_deepfm, save_checkpoint_interval: 1
2021-08-13 15:52:32,885 - INFO - **************common.configs**********
W0813 15:52:32.886739  1867 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0813 15:52:32.891144  1867 device_context.cc:422] device: 0, cuDNN Version: 7.6.
2021-08-13 15:52:37,460 - INFO - read data
2021-08-13 15:52:37,461 - INFO - reader path:criteo_reader
2021-08-13 15:52:37,461 - INFO - reader path:criteo_reader
2021-08-13 15:52:42,372 - INFO - epoch: 0, batch_id: 0, auc:0.500132, loss:[0.6964508], avg_reader_cost: 0.01165 sec, avg_batch_cost: 0.04896 sec, avg_samples: 51.20000, ips: 1045.83222 ins/s, loss: 0.696451
2021-08-13 15:55:52,142 - INFO - epoch: 0, batch_id: 100, auc:0.682062, loss:[0.50682473], avg_reader_cost: 0.00037 sec, avg_batch_cost: 1.89720 sec, avg_samples: 5120.00000, ips: 2698.71022 ins/s, loss: 0.506825
2021-08-13 15:58:56,466 - INFO - epoch: 0, batch_id: 200, auc:0.716817, loss:[0.47779626], avg_reader_cost: 0.00032 sec, avg_batch_cost: 1.84275 sec, avg_samples: 5120.00000, ips: 2778.44940 ins/s, loss: 0.477796
2021-08-13 16:02:01,150 - INFO - epoch: 0, batch_id: 300, auc:0.733832, loss:[0.4546355], avg_reader_cost: 0.00031 sec, avg_batch_cost: 1.84641 sec, avg_samples: 5120.00000, ips: 2772.95353 ins/s, loss: 0.454636
.
.
.
2021-08-13 20:11:17,361 - INFO - epoch: 0, batch_id: 8200, auc:0.795808, loss:[0.45324177], avg_reader_cost: 0.00033 sec, avg_batch_cost: 1.88111 sec, avg_samples: 5120.00000, ips: 2721.79535 ins/s, loss: 0.453242
2021-08-13 20:14:28,855 - INFO - epoch: 0, batch_id: 8300, auc:0.795979, loss:[0.4493463], avg_reader_cost: 0.00031 sec, avg_batch_cost: 1.91445 sec, avg_samples: 5120.00000, ips: 2674.40082 ins/s, loss: 0.449346
2021-08-13 20:17:35,016 - INFO - epoch: 0, batch_id: 8400, auc:0.796083, loss:[0.43260103], avg_reader_cost: 0.00040 sec, avg_batch_cost: 1.86115 sec, avg_samples: 5120.00000, ips: 2750.99465 ins/s, loss: 0.432601
2021-08-13 20:20:42,364 - INFO - epoch: 0, batch_id: 8500, auc:0.796201, loss:[0.42714283], avg_reader_cost: 0.00034 sec, avg_batch_cost: 1.87298 sec, avg_samples: 5120.00000, ips: 2733.61228 ins/s, loss: 0.427143
2021-08-13 20:23:30,465 - INFO - epoch: 0 done, auc: 0.796279,loss:[0.44033897], epoch time: 16253.00 s
2021-08-13 20:23:33,390 - INFO - Already save model in output_model_all_deepfm/0
```


- 验证日志：
```
2021-08-13 20:56:21,228 - INFO - **************common.configs**********
2021-08-13 20:56:21,228 - INFO - use_gpu: True, use_xpu: False, use_visual: False, infer_batch_size: 5120, test_data_dir: /home/aistudio/data/slot_test_data_full, start_epoch: 0, end_epoch: 4, print_interval: 100, model_load_path: output_model_all_deepfm
2021-08-13 20:56:21,228 - INFO - **************common.configs**********
W0813 20:56:21.229516 54904 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0813 20:56:21.233371 54904 device_context.cc:422] device: 0, cuDNN Version: 7.6.
2021-08-13 20:56:25,746 - INFO - read data
2021-08-13 20:56:25,747 - INFO - reader path:criteo_reader
2021-08-13 20:56:25,750 - INFO - load model epoch 0
2021-08-13 20:56:25,750 - INFO - start load model from output_model_all_deepfm/0
2021-08-13 20:56:31,285 - INFO - epoch: 0, batch_id: 0, auc: 0.790873, avg_reader_cost: 0.01153 sec, avg_batch_cost: 0.04754 sec, avg_samples: 5120.00000, ips: 92505.88 ins/s
2021-08-13 20:58:59,654 - INFO - epoch: 0, batch_id: 100, auc: 0.802344, avg_reader_cost: 0.02974 sec, avg_batch_cost: 1.48363 sec, avg_samples: 5120.00000, ips: 3450.89 ins/s
2021-08-13 21:01:21,692 - INFO - epoch: 0, batch_id: 200, auc: 0.802837, avg_reader_cost: 0.00072 sec, avg_batch_cost: 1.42019 sec, avg_samples: 5120.00000, ips: 3605.05 ins/s
2021-08-13 21:03:43,941 - INFO - epoch: 0, batch_id: 300, auc: 0.802896, avg_reader_cost: 0.03356 sec, avg_batch_cost: 1.42232 sec, avg_samples: 5120.00000, ips: 3599.65 ins/s
2021-08-13 21:05:02,232 - INFO - epoch: 0 done, auc: 0.802757, epoch time: 516.48 s
```


```python
!cat ~/data/slot_train_data_full/part-* | wc -l
# # 44000000
!cat ~/data/slot_test_data_full/part-22* | wc -l
# # 1840617
```


```python
# 动态图训练
!python -u ../../../tools/trainer.py -m config_bigdata.yaml
```


```python
# 动态图预测
!python -u ../../../tools/infer.py -m config_bigdata.yaml
```


```python

```
