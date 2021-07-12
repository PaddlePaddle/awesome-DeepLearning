# BERT 蒸馏 [[English](./README_en.md)]

## 依赖模块

* python3
* paddlepaddle-gpu==2.0.0.post101
* paddlenlp==2.0.0rc0
* paddleslim使用develop版本

## 项目介绍

```
cd ./PaddleSlim-develop/demo/ofa/bert/
|-run_glue_ofa.py：启动训练
|-export_model.py：导出相应的子模型并转为静态图模型
```

本项目支持对TinyBERT模型以DynaBERT的宽度自适应策略进行蒸馏。当前项目仅支持在 PaddleSlim-develop 版本进行运行，随后续发版会进行版本更新。

### 模型介绍

TinyBERT是由华中科技大学和华为诺亚方舟实验室在2019年联合提出的一种针对transformer-based模型的知识蒸馏方法，以BERT为例对大型预训练进行研究。TinyBERT主要进行了以下两点创新：[论文地址](https://arxiv.org/pdf/1909.10351.pdf)

1. 提供一种新的针对 transformer-based 模型进行蒸馏的方法，使得BERT中具有的语言知识可以迁移到TinyBERT中去。
2. 提出一个两阶段学习框架，在预训练阶段和fine-tuning阶段都进行蒸馏，确保TinyBERT可以充分的从BERT中学习到一般领域和特定任务两部分的知识。

DynaBERT模型是华为和北京大学在2020年联合提出的，该模型可以通过自适应宽度和深度的选择来灵活地调整网络大小，从而得到一个尺寸可变的网络。在实际工作中，不同任务需求的网络尺寸不同。该网络可以通过一次训练产生多尺寸网络从而解决不同工作需求的适配问题。

DynaBERT 的训练分为两部分，首先通过知识蒸馏的方法将 teacher BERT 的知识迁移到有自适应宽度的子网络 student $DynaBERT_w$ 中，然后再对 $DynaBERT_w$ 进行知识蒸馏得到同时支持深度自适应和宽度自适应的子网络 DynaBERT。[论文地址](https://arxiv.org/pdf/2004.04037.pdf)



## 模型训练

以 GLUE/QQP 任务为例。

```shell
export CUDA_VISIBLE_DEVICES=1
export TASK_NAME='QQP'

python -u ./run_glue_ofa.py --model_type bert \
                         --model_name_or_path ${PATH_OF_QQP} \
                         --task_name $TASK_NAME --max_seq_length 128     \
                         --batch_size 32       \
                         --learning_rate 2e-5     \
                         --num_train_epochs 6     \
                         --logging_steps 10     \
                         --save_steps 500     \
                         --output_dir ./tmp/$TASK_NAME/ \
                         --n_gpu 1 \
                         --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5
```

其中参数释义如下：

- `model_type` 指示了模型类型，当前仅支持BERT模型。
- `model_name_or_path` 预训练模型的存储地址。如果你想使用TinyBERT在QQP任务下的预训练模型，请查看：[BERT蒸馏](https://aistudio.baidu.com/aistudio/projectdetail/2177549)
- `task_name` 下游任务名称。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。默认：128。
- `batch_size` 每次迭代每张卡上的样本数目。
- `learning_rate` 学习率。
- `num_train_epochs` 训练轮数。
- `logging_steps` 日志打印间隔。
- `save_steps` 模型保存及评估间隔。
- `output_dir` 模型保存路径。
- `n_gpu` 表示使用的 GPU 卡数。若希望多卡训练，将其设置为指定数目即可；若为0，则使用CPU。
- `width_mult_list` 表示压缩训练过程中，对每层 Transformer Block 的宽度选择的范围。



## 子模型导出

根据传入的 config 导出相应的子模型并转为静态图模型。

```shell
python3.7 -u ./export_model.py --model_type bert \
                             --model_name_or_path ${PATH_OF_QQP_MODEL_AFTER_OFA} \
                             --max_seq_length 128     \
                             --sub_model_output_dir ./tmp/$TASK_NAME/dynamic_model \
                             --static_sub_model ./tmp/$TASK_NAME/static_model \
                             --n_gpu 1 \
                             --width_mult  0.6666666666666666
```

其中参数释义如下：

- `model_type` 指示了模型类型，当前仅支持BERT模型。
- `model_name_or_path` 指示了某种特定配置的经过OFA训练后保存的模型，对应有其预训练模型和预训练时使用的tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。默认：128.
- `sub_model_output_dir` 指示了导出子模型动态图参数的目录。
- `static_sub_model` 指示了导出子模型静态图模型及参数的目录，设置为None，则表示不导出静态图模型。默认：None。
- `n_gpu` 表示使用的 GPU 卡数。若希望使用多卡训练，将其设置为指定数目即可；若为0，则使用CPU。默认：1.
- `width_mult` 表示导出子模型的宽度。默认：1.0.
