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

import copy
import argparse
from tqdm import tqdm
from functools import partial
import paddle
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.metrics import AccuracyAndF1
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepModel, SkepTokenizer
from model import SkepForSquenceClassification
from utils.data import convert_example_to_feature

def evaluate(model, data_loader, metric):

    model.eval()
    metric.reset()
    for idx, batch_data in tqdm(enumerate(data_loader)):
        input_ids, token_type_ids, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)        

        # count metric
        correct = metric.compute(logits, labels)
        metric.update(correct)

    accuracy, precision, recall, f1, _ = metric.accumulate()

    return accuracy, precision, recall, f1


if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument('--test_set', choices=['dev', 'test'], default="test", help="select test set in [dev, test]")
    parser.add_argument("--batch_size", type=int, default=32, help="total examples' number in batch for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="number of words of the longest seqence.")
    args = parser.parse_args()
    # yapf: enbale

    # load dev data
    model_name = "skep_ernie_1.0_large_ch"
    test_ds = load_dataset("chnsenticorp", splits=[args.test_set])

    tokenizer = SkepTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    test_ds = test_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="float32")
    ): fn(samples)
    test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds, batch_sampler=test_batch_sampler, collate_fn=batchify_fn)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    ernie = SkepModel.from_pretrained(model_name)
    model = SkepForSquenceClassification(ernie, num_classes=len(test_ds.label_list))    
    model.load_dict(loaded_state_dict)

    metric = AccuracyAndF1()
 
    # evalute on dev data
    accuracy, precision, recall, f1  = evaluate(model, test_loader,  metric)
    print(f'evalution result: accuracy: {accuracy}, precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')
    
