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
import ast
import argparse
import warnings
from functools import partial
from data import read, load_dict, convert_example_to_features
from model import EventExtractionModel
from utils import set_seed

from evaluate import evaluate

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, LinearDecayWithWarmup
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.metrics import ChunkEvaluator

warnings.filterwarnings("ignore")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_name", type=str, default="trigger", help="The trigger or role model which you wanna train")
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--train_path", type=str, default=None, help="train data")
parser.add_argument("--dev_path", type=str, default=None, help="dev data")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.2, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--eval_step", type=int, default=100, help="evaluation step")
parser.add_argument("--log_step", type=int, default=20, help="log step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable



def train():
    # set running envir
    paddle.set_device(args.device)
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)


    model_name = "ernie-1.0"
    
    # load and process data
    tag2id, id2tag = load_dict(args.tag_path)
    train_ds = load_dataset(read, data_path=args.train_path, lazy=False)
    dev_ds = load_dataset(read, data_path=args.dev_path, lazy=False)

    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_features, tokenizer=tokenizer, tag2id=tag2id, max_seq_length=args.max_seq_len, pad_default_tag="O", is_test=False)

    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id), # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id), # token_type
        Stack(), # seq len
        Pad(axis=0, pad_val=-1) # tag_ids
    ): fn(samples)
    
    train_batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.DistributedBatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    train_loader = paddle.io.DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn)

    # configure model training     
    ernie = ErnieModel.from_pretrained(model_name)
    event_model = EventExtractionModel(ernie, num_classes=len(tag2id))

    set_seed(args.seed)    

    num_training_steps = len(train_loader) * args.num_epoch
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)
    decay_params = [p.name for n, p in event_model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=event_model.parameters(), weight_decay=args.weight_decay, apply_decay_param_fun=lambda x: x in decay_params)
 
    metric = ChunkEvaluator(label_list=tag2id.keys(), suffix=False)
   
    # start to train event_model
    global_step, best_f1 = 0,  0.
    event_model.train()
    for epoch in range(1, args.num_epoch+1):
        for batch_data in train_loader:
            input_ids, token_type_ids, seq_lens, tag_ids = batch_data
            logits = event_model(input_ids, token_type_ids)
            loss = event_model.get_crf_loss(logits, seq_lens, tag_ids)            

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()
 
            if global_step > 0 and global_step % args.log_step == 0 and rank == 0:
                print(f"{args.model_name} - epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}") 
            if global_step > 0 and global_step % args.eval_step == 0 and rank == 0: 
                precision, recall, f1_score = evaluate(event_model, dev_loader, metric)
                event_model.train()
                if f1_score > best_f1:
                    print(f"best F1 performence has been updated: {best_f1:.5f} --> {f1_score:.5f}")
                    best_f1 = f1_score
                    paddle.save(event_model.state_dict(), f"{args.checkpoint}/{args.model_name}_best.pdparams")
                print(f'{args.model_name} evalution result: precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1_score:.5f} current best {best_f1:.5f}')
            global_step += 1

    if rank == 0:
        paddle.save(event_model.state_dict(), f"{args.checkpoint}/{args.model_name}_final.pdparams")   
       

if __name__=="__main__":
    train()
