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

本项目基于预训练模型ERNIE进行中文阅读理解，使用的数据集是Dureader_robust数据集。

### 模型介绍

ERINE是百度发布一个预训练模型，它通过引入三种级别的Knowledge Masking帮助模型学习语言知识，在多项任务上超越了BERT。


## 模型训练

```shell
export CUDA_VISIBLE_DEVICES=0

python ./train.py --model_name ernie-1.0 \
                         --epochs 1       \
                         --learning_rate 3e-5     \
                         --max_seq_length 512     \
                         --batch_size 12     \
                         --warmup_proportion 0.1 \
                         --weight_decay 0.01 \
                         --save_model_path ./ernie_rc.pdparams \
                         --save_opt_path ./ernie_rc.pdopt
```

其中参数释义如下：

- `model_name` 需要加载的模型名字。
- `epochs` 训练轮次。
- `learning_rate` 学习率。
- `max_seq_length` 最大句子长度，超过将会被截断。
- `batch_size` 每次迭代每张卡上的样本数目。
- `warmup_proportion` warmup占据总的训练迭代次数的比例。
- `weight_decay` 权重衰减值。
- `save_model_path` 模型保存路径。
- `save_opt_path` 优化器保存路径。

## 模型评估

运行evaluate.py脚本进行模型评估。

```shell
export CUDA_VISIBLE_DEVICES=0

python ./evaluate.py --model_path ./ernie_rc.pdparams \
                             --max_seq_length 512     \
                             --batch_size 12 
```