# 基于ERNIE的阅读理解

## 依赖模块

* python3
* paddlepaddle-gpu==2.0.0.post101
* paddlenlp==2.0.1

## 项目介绍

```
|-data_proessor.py：数据处理相关代码
|-train.py：模型训练代码
|-evaluate.py：模型评估代码
|-utilis.py：定义模型训练时用到的一些组件
```

本项目基于预训练模型XLNet进行中文阅读理解，使用的数据集是IMDB数据集。

### 模型介绍

XLNet整体上是基于自回归模型的建模思路设计的，同时避免了只能单向建模的缺点，因此它是一种能看得见双向信息的广义AR模型。

## 模型训练

```shell
export CUDA_VISIBLE_DEVICES=0

python ./train.py --model_name_or_path xlnet-base-cased \
                         --task_name sst-2 \
                         --num_train_epochs 1       \
                         --learning_rate 2e-5     \
                         --max_seq_length 128     \
                         --batch_size 32     \
                         --warmup_proportion 0.1 \
                         --weight_decay 0.0 \
                         --logging_steps 100 \
                         --save_steps 500 \
                         --output_dir ./tmp
```

其中参数释义如下：

- `model_name_or_path` 需要加载的模型名字。
- `task_name` 训练轮次。
- `num_train_epochs` 训练轮次。
- `learning_rate` 学习率。
- `max_seq_length` 最大句子长度，超过将会被截断。
- `batch_size` 每次迭代每张卡上的样本数目。
- `warmup_proportion` warmup占据总的训练迭代次数的比例。
- `weight_decay` 权重衰减值。
- `logging_steps` 多少步打印一次日志。
- `save_steps` 多少步保存一次模型。
- `output_dir` 输出目录。
