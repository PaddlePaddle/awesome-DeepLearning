
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

import numpy as np
from functools import partial
import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddlenlp.data import Vocab, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.datasets import load_dataset

def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def read(src_path, tgt_path, is_predict=False):
    if is_predict:
        with open(src_path, 'r', encoding='utf8') as src_f:
            for src_line in src_f.readlines():
                src_line = src_line.strip()
                if not src_line:
                    continue
                yield {'src':src_line, 'tgt':''}
    else:
        with open(src_path, 'r', encoding='utf8') as src_f, open(tgt_path, 'r', encoding='utf8') as tgt_f:
            for src_line, tgt_line in zip(src_f.readlines(), tgt_f.readlines()):
                src_line = src_line.strip()
                if not src_line:
                    continue
                tgt_line = tgt_line.strip()
                if not tgt_line:
                    continue
                yield {'src':src_line, 'tgt':tgt_line}

def prepare_train_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])
    trg_word = word_pad([[bos_idx] + inst[1] for inst in insts])
    lbl_word = np.expand_dims(
        word_pad([inst[1] + [eos_idx] for inst in insts]), axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs


# 创建训练集、验证集的dataloader
def create_data_loader(args):
    train_dataset = load_dataset(read, src_path=args.training_file.split(',')[0], tgt_path=args.training_file.split(',')[1], lazy=False)
    dev_dataset = load_dataset(read, src_path=args.training_file.split(',')[0], tgt_path=args.training_file.split(',')[1], lazy=False)
    print('load src vocab')
    print( args.src_vocab_fpath)
    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    print('load trg vocab')
    print(args.trg_vocab_fpath)
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    print('padding')
    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))
    print('convert example')
    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    data_loaders = [(None)] * 2
    print('dataset loop')
    for i, dataset in enumerate([train_dataset, dev_dataset]):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(
                min_max_filer, max_len=args.max_length))

        sampler = SamplerHelper(dataset)

        if args.sort_type == SortType.GLOBAL:
            src_key = (lambda x, data_source: len(data_source[x][0]) + 1)
            trg_key = (lambda x, data_source: len(data_source[x][1]) + 1)
            # Sort twice
            sampler = sampler.sort(key=trg_key).sort(key=src_key)
        else:
            if args.shuffle:
                sampler = sampler.shuffle(seed=args.shuffle_seed)
            max_key = (lambda x, data_source: max(len(data_source[x][0]), len(data_source[x][1])) + 1)
            if args.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=args.pool_size)

        batch_size_fn = lambda new, count, sofar, data_source: max(sofar, len(data_source[new][0]) + 1,
                                                                   len(data_source[new][1]) + 1)
        batch_sampler = sampler.batch(
            batch_size=args.batch_size,
            drop_last=False,
            batch_size_fn=batch_size_fn,
            key=lambda size_so_far, minibatch_len: size_so_far * minibatch_len)

        if args.shuffle_batch:
            batch_sampler = batch_sampler.shuffle(seed=args.shuffle_seed)

        if i == 0:
            batch_sampler = batch_sampler.shard()

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                pad_idx=args.bos_idx),
            num_workers=2,
            return_list=True)
        data_loaders[i] = (data_loader)
    return data_loaders

class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"