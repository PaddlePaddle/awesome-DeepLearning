#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT fine-tuning in Paddle Dygraph Mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import ast
import time
import argparse
import numpy as np
import multiprocessing
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, Layer
from .reader.cls import *
from .model.bert import BertConfig
from .model.cls import ClsModelLayer
from .optimization import Optimizer
from .utils.init import init_from_static_model

__all__ = ["BERTClassifier"]


def create_data(batch):
    """
    convert data to variable
    """
    src_ids = to_variable(batch[0], "src_ids")
    position_ids = to_variable(batch[1], "position_ids")
    sentence_ids = to_variable(batch[2], "sentence_ids")
    input_mask = to_variable(batch[3], "input_mask")
    labels = to_variable(batch[4], "labels")
    labels.stop_gradient = True
    return src_ids, position_ids, sentence_ids, input_mask, labels


class BERTClassifier(Layer):
    def __init__(self,
                 num_labels,
                 task_name="mnli",
                 model_path=None,
                 use_cuda=True,
                 return_pooled_out=True):
        super(BERTClassifier, self).__init__()
        self.task_name = task_name.lower()
        BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12/"
        bert_config_path = BERT_BASE_PATH + "/bert_config.json"
        self.vocab_path = BERT_BASE_PATH + "/vocab.txt"
        self.init_pretraining_params = BERT_BASE_PATH + "/dygraph_params/"
        self.do_lower_case = True
        self.bert_config = BertConfig(bert_config_path)

        if use_cuda:
            self.dev_count = fluid.core.get_cuda_device_count()
        else:
            self.dev_count = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        self.trainer_count = fluid.dygraph.parallel.Env().nranks

        self.processors = {
            'xnli': XnliProcessor,
            'cola': ColaProcessor,
            'mrpc': MrpcProcessor,
            'mnli': MnliProcessor,
        }

        self.cls_model = ClsModelLayer(
            self.bert_config, num_labels, return_pooled_out=return_pooled_out)

        if model_path is not None:
            #restore the model
            print("Load params from %s" % model_path)
            model_dict, _ = fluid.load_dygraph(model_path)
            self.cls_model.load_dict(model_dict)
        elif self.init_pretraining_params:
            print("Load pre-trained model from %s" %
                  self.init_pretraining_params)
            init_from_static_model(self.init_pretraining_params, self.cls_model,
                                   self.bert_config)
        else:
            raise Exception(
                "You should load pretrained model for training this teacher model."
            )

    def emb_names(self):
        return self.cls_model.emb_names()

    def forward(self, input):
        return self.cls_model(input)

    def test(self, data_dir, batch_size=64, max_seq_len=512):

        processor = self.processors[self.task_name](
            data_dir=data_dir,
            vocab_path=self.vocab_path,
            max_seq_len=max_seq_len,
            do_lower_case=self.do_lower_case,
            in_tokens=False)

        test_data_generator = processor.data_generator(
            batch_size=batch_size, phase='dev', epoch=1, shuffle=False)

        self.cls_model.eval()
        total_cost, final_acc, avg_acc, total_num_seqs = [], [], [], []
        for batch in test_data_generator():
            data_ids = create_data(batch)

            total_loss, _, _, np_acces, np_num_seqs = self.cls_model(data_ids)

            np_loss = total_loss.numpy()
            np_acc = np_acces[-1].numpy()
            np_avg_acc = np.mean([acc.numpy() for acc in np_acces])
            np_num_seqs = np_num_seqs.numpy()

            total_cost.extend(np_loss * np_num_seqs)
            final_acc.extend(np_acc * np_num_seqs)
            avg_acc.extend(np_avg_acc * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)

        print("[evaluation] classifier[-1] average acc: %f; average acc: %f" %
              (np.sum(final_acc) / np.sum(total_num_seqs),
               np.sum(avg_acc) / np.sum(total_num_seqs)))
        self.cls_model.train()

    def fit(self,
            data_dir,
            epoch,
            batch_size=64,
            use_cuda=True,
            max_seq_len=512,
            warmup_proportion=0.1,
            use_data_parallel=False,
            learning_rate=0.00005,
            weight_decay=0.01,
            lr_scheduler="linear_warmup_decay",
            skip_steps=10,
            save_steps=1000,
            checkpoints="checkpoints"):

        processor = self.processors[self.task_name](
            data_dir=data_dir,
            vocab_path=self.vocab_path,
            max_seq_len=max_seq_len,
            do_lower_case=self.do_lower_case,
            in_tokens=False,
            random_seed=5512)
        shuffle_seed = 1 if self.trainer_count > 1 else None

        train_data_generator = processor.data_generator(
            batch_size=batch_size,
            phase='train',
            epoch=epoch,
            dev_count=self.trainer_count,
            shuffle=True,
            shuffle_seed=shuffle_seed)
        num_train_examples = processor.get_num_examples(phase='train')
        max_train_steps = epoch * num_train_examples // batch_size // self.trainer_count
        warmup_steps = int(max_train_steps * warmup_proportion)

        print("Device count: %d" % self.dev_count)
        print("Trainer count: %d" % self.trainer_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        if use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        optimizer = Optimizer(
            warmup_steps=warmup_steps,
            num_train_steps=max_train_steps,
            learning_rate=learning_rate,
            model_cls=self.cls_model,
            weight_decay=weight_decay,
            scheduler=lr_scheduler,
            loss_scaling=1.0,
            parameter_list=self.cls_model.parameters())

        if use_data_parallel:
            self.cls_model = fluid.dygraph.parallel.DataParallel(self.cls_model,
                                                                 strategy)
            train_data_generator = fluid.contrib.reader.distributed_batch_reader(
                train_data_generator)

        steps = 0
        time_begin = time.time()

        for batch in train_data_generator():
            data_ids = create_data(batch)
            total_loss, logits, losses, accuracys, num_seqs = self.cls_model(
                data_ids)

            optimizer.optimization(
                total_loss,
                use_data_parallel=use_data_parallel,
                model=self.cls_model)
            self.cls_model.clear_gradients()

            if steps != 0 and steps % skip_steps == 0:
                time_end = time.time()
                used_time = time_end - time_begin
                current_example, current_epoch = processor.get_train_progress()
                localtime = time.asctime(time.localtime(time.time()))
                print(
                    "%s, epoch: %s, steps: %s, dy_graph loss: %f, acc: %f, speed: %f steps/s"
                    % (localtime, current_epoch, steps, total_loss.numpy(),
                       accuracys[-1].numpy(), skip_steps / used_time))
                time_begin = time.time()

            if steps != 0 and steps % save_steps == 0 and fluid.dygraph.parallel.Env(
            ).local_rank == 0:

                self.test(data_dir, batch_size=64, max_seq_len=512)

                save_path = os.path.join(checkpoints,
                                         "steps" + "_" + str(steps))
                fluid.save_dygraph(self.cls_model.state_dict(), save_path)
                fluid.save_dygraph(optimizer.optimizer.state_dict(), save_path)
                print("Save model parameters and optimizer status at %s" %
                      save_path)

            steps += 1

        if fluid.dygraph.parallel.Env().local_rank == 0:
            save_path = os.path.join(checkpoints, "final")
            fluid.save_dygraph(self.cls_model.state_dict(), save_path)
            fluid.save_dygraph(optimizer.optimizer.state_dict(), save_path)
            print("Save model parameters and optimizer status at %s" %
                  save_path)
