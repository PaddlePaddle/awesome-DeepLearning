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
from paddlenlp.data import Stack, Pad, Tuple

from model import JointModel
from data import read, load_dict, convert_example_to_feature
from metric import SeqEntityScore, MultiLabelClassificationScore



def evaluate(joint_model, data_loader, intent_metric, slot_metric):

    joint_model.eval()
    intent_metric.reset()
    slot_metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, intent_labels, tag_ids, history_ids = batch_data
        intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids, history_ids=history_ids)
        # count intent metric
        intent_metric.update(pred_labels=intent_logits, real_labels=intent_labels)
        # count slot metric
        slot_pred_labels = slot_logits.argmax(axis=-1)
        slot_metric.update(pred_paths=slot_pred_labels, real_paths=tag_ids)

    intent_results = intent_metric.get_result()
    slot_results = slot_metric.get_result()

    return intent_results, slot_results


if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument("--test_path", type=str, default=None, help="test data")
    parser.add_argument("--slot_dict_path", type=str, default=None, help="slot dict path")
    parser.add_argument("--intent_dict_path", type=str, default=None, help="intent dict path")
    parser.add_argument("--use_history", type=bool, default=False, help="use history or not")    
    parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
    args = parser.parse_args()
    # yapf: enbale

    # load dev data
    model_name = "ernie-1.0"
    intent2id, id2intent = load_dict(args.intent_dict_path)
    slot2id, id2slot = load_dict(args.slot_dict_path)
    test_ds = load_dataset(read, data_path=args.test_path, lazy=False)
    
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, slot2id=slot2id, intent2id=intent2id, use_history=args.use_history, pad_default_tag="O", max_seq_len=args.max_seq_len)
    test_ds = test_ds.map(trans_func, lazy=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype="float32"),
        Pad(axis=0, pad_val=slot2id["O"], dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_id)
    ):fn(samples)

    batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds, batch_sampler=batch_sampler, collate_fn=batchify_fn)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    ernie = ErnieModel.from_pretrained(model_name)
    joint_model = JointModel(ernie, len(slot2id), len(intent2id), use_history=args.use_history, dropout=0.1)    
    joint_model.load_dict(loaded_state_dict)

    print(args.use_history)
    intent_metric = MultiLabelClassificationScore(id2intent)
    slot_metric = SeqEntityScore(id2slot)
 
    # evalute on dev data
    intent_results, slot_results = evaluate(joint_model, test_loader, intent_metric, slot_metric)
    intent_result, slot_result = intent_results["Total"], slot_results["Total"]
    print(f'intent evalution result: precision: {intent_result["Precision"]:.5f}, recall: {intent_result["Recall"]:.5f},  F1: {intent_result["F1"]:.5f}')
    print(f'slot evalution result: precision: {slot_result["Precision"]:.5f}, recall: {slot_result["Recall"]:.5f},  F1: {slot_result["F1"]:.5f}')
    


