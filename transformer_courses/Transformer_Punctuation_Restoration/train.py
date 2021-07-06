
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
 
import numpy as np
from pprint import pprint
from attrdict import AttrDict

import paddle
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, position_encoding_init
from paddlenlp.utils.log import logger
from dataloader import create_data_loader
import paddle.distributed as dist

# 模型训练
  
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
 

if __name__ == '__main__':
    # 读入参数
    global_step = 0
    logging_steps=200 # 日志的保存周期
    last_step = num_train_epochs * len(train_data_loader)
    tic_train = time.time()
    save_steps=200 # 模型保存周期
    output_dir='checkpoints/' # 模型保存目录

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, _, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = loss_fct(logits, labels)
            avg_loss = paddle.mean(loss)
            if global_step % logging_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, avg_loss,
                        logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % save_steps == 0 or global_step == last_step:
                if paddle.distributed.get_rank() == 0:
                        evaluate(model, loss_fct, test_data_loader,
                                    label_num)
                        paddle.save(model.state_dict(),os.path.join(output_dir,
                                                    "model_%d.pdparams" % global_step))

