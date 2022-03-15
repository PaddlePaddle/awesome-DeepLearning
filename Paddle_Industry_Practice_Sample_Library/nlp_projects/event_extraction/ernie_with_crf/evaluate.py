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


import argparse
import paddle
from functools import partial
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from paddlenlp.datasets import load_dataset

from model import EventExtractionModel
from data import read, load_dict, convert_example_to_features
from paddlenlp.data import Stack, Pad, Tuple



def evaluate(model, data_loader, metric):
    

    model.eval()
    metric.reset()
    for batch_data in data_loader:
        input_ids, token_type_ids, seq_lens, tag_ids = batch_data
        logits = model(input_ids, token_type_ids)
        _, pred_paths = model.viterbi_decoder(logits, seq_lens)
        #preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(seq_lens, pred_paths, tag_ids)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())  
        precision, recall, f1_score = metric.accumulate()
    
    return precision, recall, f1_score      





if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="trigger", help="The trigger or role model which you wanna evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument("--dev_path", type=str, default=None, help="dev data")
    parser.add_argument("--tag_path", type=str, default=None, help="tag dict path")
    parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")

    args = parser.parse_args()
    # yapf: enbale

    # load dev data
    model_name = "ernie-1.0"
    tag2id, id2tag = load_dict(args.tag_path)
    dev_ds = load_dataset(read, data_path=args.dev_path, lazy=False)
    
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_features, tokenizer=tokenizer, tag2id=tag2id,  max_seq_length=256, pad_default_tag="O", is_test=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id), # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id), # token_type
        Stack(), # seq len
        Pad(axis=0, pad_val=-1) # tag_ids
    ): fn(samples)

    batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_loader = paddle.io.DataLoader(dev_ds, batch_sampler=batch_sampler, collate_fn=batchify_fn)

    # load model
    model_name = "ernie-1.0"
    loaded_state_dict = paddle.load(args.model_path)
    ernie = ErnieModel.from_pretrained(model_name)
    event_model = EventExtractionModel(ernie, num_classes=len(tag2id))
    event_model.load_dict(loaded_state_dict)

    metric = ChunkEvaluator(label_list=tag2id.keys(), suffix=False)
    
    # evalute on dev data
    precision, recall, f1_score = evaluate(event_model, dev_loader, metric)
    print(f'{args.model_name} evalution result:  precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1_score:.5f}')

    


