
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import yaml 
import argparse 
from pprint import pprint
from attrdict import AttrDict

import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, position_encoding_init
from paddlenlp.utils.log import logger
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.transformers import ElectraForTokenClassification, ElectraTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict

from dataloader import create_dataloader,load_dataset
 

import paddle.distributed as dist
 
import yaml 
import argparse 
from pprint import pprint
from attrdict import AttrDict

import paddle
from paddle.io import DataLoader  

import os
import pandas as pd
from sklearn.metrics import classification_report
from functools import partial


def compute_metrics(labels, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    labels=[x for batch in labels for x in batch]
    outputs = []
    nb_correct=0
    nb_true=0
    val_f1s=[]
    label_vals=[0,1,2,3]
    y_trues=[]
    y_preds=[]
    for idx, end in enumerate(lens):
        y_true = labels[idx][:end].tolist()
        y_pred = [x for x in decodes[idx][:end]]
        nb_correct += sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
        nb_true+=len(y_true)
        y_trues.extend(y_true)
        y_preds.extend(y_pred)

    score = nb_correct / nb_true
    # val_f1 = metrics.f1_score(y_trues, y_preds, average='micro', labels=label_vals)

    result=classification_report(y_trues, y_preds)
    # print(val_f1)   
    return score,result
    
def evaluate(model, loss_fct, data_loader, label_num):
    model.eval()
    pred_list = []
    len_list = []
    labels_list=[]
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(length.numpy())
        labels_list.append(labels.numpy())
    accuracy,result=compute_metrics(labels_list, pred_list, len_list)
    print("eval loss: %f, accuracy: %f" % (avg_loss, accuracy))
    print(result)
    model.train()
 
# evaluate(model, loss_fct, metric, test_data_loader,label_num)
 
def do_train(args):
    last_step =  args.num_train_epochs * len(train_data_loader)
    tic_train = time.time()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            args.global_step += 1
            print('~~~~~~~~~~~~~~~~~~~~~args.global_step',args.global_step)
            input_ids, token_type_ids, _, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = loss_fct(logits, labels)
            avg_loss = paddle.mean(loss) 

            if args.global_step % args.logging_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (args.global_step, epoch, step, avg_loss,
                        args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if args.global_step % args.save_steps == 0 or args.global_step == last_step:
                if paddle.distributed.get_rank() == 0:
                        evaluate(model, loss_fct, test_data_loader, label_num)
                        paddle.save(model.state_dict(),os.path.join(args.output_dir,
                                                    "model_%d.pdparams" % args.global_step))

# 模型训练
if __name__ == '__main__':
 
    # 读入参数
    yaml_file = './electra.base.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        # pprint(args)
    
    paddle.set_device(args.device) # 使用gpu，相应地，安装paddlepaddle-gpu
    
    train_data_loader, test_data_loader  = create_dataloader(args)

    # 加载dataset
    # Create dataset, tokenizer and dataloader.
    train_ds, test_ds = load_dataset('TEDTalk', splits=('train', 'test'), lazy=False)
    label_list = train_ds.label_list
    label_num = len(label_list)

    # 加载预训练模型  
    # Define the model netword and its loss
    model = ElectraForTokenClassification.from_pretrained(args.model_name_or_path, num_classes= label_num)

    # 设置AdamW优化器
    num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(float(args.learning_rate), num_training_steps, args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

    optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=float(args.adam_epsilon),
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)     

    # 设置CrossEntropy损失函数 
    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=args.ignore_label)

    # 设置评估方式
    metric = paddle.metric.Accuracy()

    # 开始训练
    do_train(args)
