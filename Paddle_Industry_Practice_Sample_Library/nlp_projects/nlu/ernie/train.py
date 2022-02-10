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


import paddle
import os
import ast
import argparse
import warnings
import numpy as np
from functools import partial
from data import read, load_dict, convert_example_to_feature
from model import JointModel
from utils import set_seed
from metric import SeqEntityScore, MultiLabelClassificationScore
from evaluate import evaluate

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, LinearDecayWithWarmup
from paddlenlp.data import Stack, Pad, Tuple

warnings.filterwarnings("ignore")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--slot_dict_path", type=str, default=None, help="slot dict path")
parser.add_argument("--intent_dict_path", type=str, default=None, help="intent dict path")
parser.add_argument("--train_path", type=str, default=None, help="train data")
parser.add_argument("--dev_path", type=str, default=None, help="dev data")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max grad norm to clip gradient.")
parser.add_argument("--eval_step", type=int, default=500, help="evaluation step")
parser.add_argument("--log_step", type=int, default=50, help="log step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--use_history", type=bool, default=False, help="Use history in dataset or not")
parser.add_argument("--intent_weight", type=bool, default=True, help="Use intent weight strategy")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


class JointLoss(paddle.nn.Layer):
    def __init__(self, intent_weight=None):
        super(JointLoss, self).__init__()
        self.intent_criterion = paddle.nn.BCEWithLogitsLoss(weight=intent_weight)
        self.slot_criterion = paddle.nn.CrossEntropyLoss()

    def forward(self, intent_logits, slot_logits, intent_labels, slot_labels):
        intent_loss = self.intent_criterion(intent_logits, intent_labels)
        slot_loss = self.slot_criterion(slot_logits, slot_labels)
        loss = intent_loss + slot_loss

        return loss


def train():
    # set running envir
    paddle.set_device(args.device)
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    model_name = "ernie-1.0"

    # load and process data
    intent2id, id2intent = load_dict(args.intent_dict_path)
    slot2id, id2slot = load_dict(args.slot_dict_path)

    train_ds = load_dataset(read, data_path=args.train_path, lazy=False)
    dev_ds = load_dataset(read, data_path=args.dev_path, lazy=False)

    # compute intent weight
    if args.intent_weight:
        intent_weight = [1] * len(intent2id)
        for example in train_ds:
            for intent in example["intent_labels"]:
                intent_weight[intent2id[intent]] += 1
        for intent, intent_id in intent2id.items():
            neg_pos = (len(train_ds) - intent_weight[intent_id]) / intent_weight[intent_id]
            intent_weight[intent_id] = np.log10(neg_pos)
        intent_weight = paddle.to_tensor(intent_weight)

    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, slot2id=slot2id, intent2id=intent2id, use_history=args.use_history, pad_default_tag="O", max_seq_len=args.max_seq_len)

    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id), 
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="float32"),
        Pad(axis=0, pad_val=slot2id["O"], dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_id)
    ):fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    train_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn, return_list=True)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn, return_list=True)


    # configure model training
    ernie = ErnieModel.from_pretrained(model_name)
    joint_model = JointModel(ernie, len(slot2id), len(intent2id), use_history=args.use_history, dropout=0.1)

    num_training_steps = len(train_loader) * args.num_epoch
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)
    decay_params = [p.name for n, p in joint_model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=joint_model.parameters(), weight_decay=args.weight_decay, apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

    if args.intent_weight:
        joint_loss = JointLoss(intent_weight)
    else:
        joint_loss = JointLoss()

    intent_metric = MultiLabelClassificationScore(id2intent)
    slot_metric = SeqEntityScore(id2slot)
    # start to train joint_model
    global_step, intent_best_f1, slot_best_f1 = 0, 0., 0.
    joint_model.train() 
    for epoch in range(1, args.num_epoch+1):
        for batch_data in train_loader:
            input_ids, token_type_ids, intent_labels, tag_ids, history_ids = batch_data
            intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids, history_ids=history_ids)

            loss = joint_loss(intent_logits, slot_logits, intent_labels, tag_ids)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % args.log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if global_step > 0 and global_step % args.eval_step == 0:
                intent_results, slot_results = evaluate(joint_model, dev_loader, intent_metric, slot_metric)
                intent_result, slot_result = intent_results["Total"], slot_results["Total"]
                joint_model.train()
                intent_f1, slot_f1 = intent_result["F1"], slot_result["F1"]
                if intent_f1 > intent_best_f1 or slot_f1 > slot_best_f1:
                    paddle.save(joint_model.state_dict(), f"{args.checkpoint}/best.pdparams")
                if intent_f1 > intent_best_f1:
                    print(f"intent best F1 performence has been updated: {intent_best_f1:.5f} --> {intent_f1:.5f}")
                    intent_best_f1 = intent_f1
                if slot_f1 > slot_best_f1:
                    print(f"slot best F1 performence has been updated: {slot_best_f1:.5f} --> {slot_f1:.5f}")
                    slot_best_f1 = slot_f1
                print(f'intent evalution result: precision: {intent_result["Precision"]:.5f}, recall: {intent_result["Recall"]:.5f},  F1: {intent_result["F1"]:.5f}, current best {intent_best_f1:.5f}')
                print(f'slot evalution result: precision: {slot_result["Precision"]:.5f}, recall: {slot_result["Recall"]:.5f},  F1: {slot_result["F1"]:.5f}, current best {slot_best_f1:.5f}\n')

            global_step += 1



if __name__=="__main__":
    train()
