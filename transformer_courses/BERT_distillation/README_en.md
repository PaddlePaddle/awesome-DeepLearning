# BERT Distillation [[简体中文](./README.md)]

## Dependent packages

* python3
* paddlepaddle-gpu==2.0.0.post101
* paddlenlp==2.0.0rc0
* paddleslim develop version

## Project Introduction

```
cd ./PaddleSlim-develop/demo/ofa/bert/
|-run_glue_ofa.py：Start training
|-export_model.py：Export sub-model and turn it into a static graph model
```

This project supports TinyBERT distillation using DynaBERT's width adaptive strategy. The current project only supports running in the PaddleSlim-develop version, and we will update with subsequent releases.

### Model introduction

TinyBERT is a transformer model-based knowledge distillation method jointly proposed by Huazhong University of Science and Technology and Huawei Noah's Ark Lab in 2019. It uses BERT as an example to study large-scale pre-training model.

The main contributions of this work are as follows: [Paper](https://arxiv.org/pdf/1909.10351.pdf)

1. They propose a new Transformer distillation method to encourage that the linguistic knowledge encoded in teacher BERT can be adequately transferred to TinyBERT;
2. We propose a novel two-stage learning framework with performing the proposed Transformer distillation at both the pre-training and fine-tuning stages, which ensures that TinyBERT can absorb both the general-domain and task-specific knowledge of the teacher BERT.

DynaBERT is jointly proposed by Huawei Noah’s Ark Lab and Peking University in 2020. The model can flexibly adjust the network size through the choice of adaptive width and depth. In actual work, different tasks require different network sizes. This network can generate a multi-size model through one training to solve the adaptation problem of different work requirements.

The training of DynaBERT is divided into two parts. Firstly, teacher BERT 's knowledge is transferred to the sub-network student $DynaBERT_w$ with adaptive width, and then the knowledge distillation is performed on $DynaBERT_w$ to train DynaBERT with adaptive width and depth. [Paper](https://arxiv.org/pdf/2004.04037.pdf)

## Model training

Take the GLUE/QQP task as an example.

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

The parameter definitions are as follows :

* `model_type` indicates model type, currently only the BERT model is supported.
* `model_name_or_path` path to load a pre-trained model. If you want to use TinyBERT's pre-trained model under the QQP task, please check: [BERT Distillation](https://aistudio.baidu.com/aistudio/projectdetail/2177549)
* `task_name` downstream task name.
* `max_seq_length` Indicates the maximum sentence length, beyond which will be truncated. Default: 128.
* `batch_size` batch size.
* `learning_rate` learning rate.
* `num_train_epochs` number of training epochs.
* `logging_steps` log printing steps.
* `save_steps` model saving steps.
* `output_dir` model saving path.
* `n_gpu` number of gpu.
* `width_mult_list` indicates the range of selection for the width of each layer of Transformer Block during compression training.



## Sub-model export

According to the config, the corresponding sub-model is exported and converted into a static graph model.

```shell
python3.7 -u ./export_model.py --model_type bert \
                             --model_name_or_path ${PATH_OF_QQP_MODEL_AFTER_OFA} \
                             --max_seq_length 128     \
                             --sub_model_output_dir ./tmp/$TASK_NAME/dynamic_model \
                             --static_sub_model ./tmp/$TASK_NAME/static_model \
                             --n_gpu 1 \
                             --width_mult  0.6666666666666666
```

The parameter definitions are as follows :

* `model_type` indicates model type, currently only the BERT model is supported.
* `model_name_or_path` path to load model after OFA training.
* `max_seq_length` indicates the maximum sentence length, beyond which will be truncated. Default: 128.
* `sub_model_output_dir` indicates the directory for exporting the dynamic graph model and parameters of sub-model.
* `static_sub_model` indicates the directory for exporting the static graph model and parameters of sub-model. Set to None, it means that the static graph model is not exported. Default: None.
* `n_gpu` number of gpu.
* `width_mult` indicates the width of the exported sub-model. Default: 1.0
