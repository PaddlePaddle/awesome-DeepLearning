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
import copy
import argparse
import warnings
import numpy as np
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import SkepTokenizer, SkepModel, LinearDecayWithWarmup
from model import SkepForSquenceClassification
from utils.utils import set_seed
from utils.data import convert_example_to_feature

warnings.filterwarnings("ignore")


# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max grad norm to clip gradient.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--log_step", type=int, default=50, help="log step")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--checkpoint", type=str, default=None, help="Directory to model checkpoint")

args = parser.parse_args()
# yapf: enable


def train():
    # set running envir
    model_name = "skep_ernie_1.0_large_ch"

    paddle.set_device(args.device)
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    # load and process data
    train_ds = load_dataset("seabsa16", "phns", splits=["train"])

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    train_ds = train_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="int64")
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    train_loader = paddle.io.DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn)

    # configure model training
    skep = SkepModel.from_pretrained(model_name)
    model = SkepForSquenceClassification(skep, num_classes=len(train_ds.label_list))
    
    num_training_steps = len(train_loader) * args.num_epoch
    lr_scheduler = LinearDecayWithWarmup(learning_rate=args.learning_rate, total_steps=num_training_steps, warmup=args.warmup_proportion)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=args.weight_decay, apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

    # start to train model
    global_step = 1
    model.train()
    for epoch in range(1, args.num_epoch+1):
        for batch_data in train_loader():
            input_ids, token_type_ids, labels = batch_data
            logits = model(input_ids, token_type_ids=token_type_ids)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % args.log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            
            global_step += 1

    paddle.save(model.state_dict(), f"{args.checkpoint}/final.pdparams")


if __name__=="__main__":
    train()




    
