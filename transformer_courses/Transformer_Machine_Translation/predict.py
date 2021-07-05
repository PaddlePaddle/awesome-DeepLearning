
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
import logging
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict

import numpy as np
from functools import partial
import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddlenlp.data import Vocab, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, position_encoding_init
from paddlenlp.utils.log import logger


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

# 创建测试集的dataloader
def create_infer_loader(args):
    dataset = load_dataset(read, src_path=args.predict_file, tgt_path=None, is_predict=True, lazy=False)

    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])

    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    dataset = dataset.map(convert_samples, lazy=False)

    batch_sampler = SamplerHelper(dataset).batch(
        batch_size=args.infer_batch_size, drop_last=False)

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_infer_input,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            pad_idx=args.bos_idx),
        num_workers=2,
        return_list=True)
    return data_loader, trg_vocab.to_tokens


def prepare_infer_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])

    return [src_word, ]

def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(args):
    if args.use_gpu:
        place = "gpu:0"
    else:
        place = "cpu"

    paddle.set_device(place)

    # Define data loader
    test_loader, to_tokens = create_infer_loader(args)

    # Define model
    transformer = InferTransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len)

    # Load the trained model
    # assert args.init_from_params, (
    #     "Please set init_from_params to load the infer model.")
    init_from_params='trained_models/step_final'
    model_dict = paddle.load(
        os.path.join(init_from_params, "transformer.pdparams"))

    # To avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    transformer.load_dict(model_dict)

    # Set evaluate mode
    transformer.eval()

    f = open(args.output_file, "w")
    with paddle.no_grad():
        for (src_word, ) in test_loader:
            finished_seq = transformer(src_word=src_word)
            finished_seq = finished_seq.numpy().transpose([0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best:
                        break
                    id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                    word_list = to_tokens(id_list)
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence)
    f.close()

if __name__ == '__main__':
    # 读入参数
    yaml_file = './transformer.base.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_predict(args)