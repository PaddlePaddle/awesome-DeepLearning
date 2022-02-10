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
from functools import partial
import paddle
from paddlenlp.transformers import SkepModel, SkepTokenizer
from model import SkepForSquenceClassification


def predict(text, text_pair, model, tokenizer, id2label, max_seq_len=256):

    model.eval()
        
    # processing input text
    encoded_inputs = tokenizer(text=text, text_pair=text_pair, max_seq_len=max_seq_len)
    input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
    token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

    # predict by model and decoding result 
    logits = model(input_ids, token_type_ids=token_type_ids)
    label_id =  paddle.argmax(logits, axis=1).numpy()[0]

    # print predict result
    print(f"text: {text} \ntext_pair:{text_pair} \nlabel: {id2label[label_id]}")


if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument("--max_seq_len", type=int, default=512, help="number of words of the longest seqence.")
    args = parser.parse_args()
    # yapf: enbale

    # process data related
    model_name = "skep_ernie_1.0_large_ch"
    id2label = {0:"Negative", 1:"Positive"}

    tokenizer = SkepTokenizer.from_pretrained(model_name)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    ernie = SkepModel.from_pretrained(model_name)
    model = SkepForSquenceClassification(ernie, num_classes=2)    
    model.load_dict(loaded_state_dict)
 
    # predict with model
    text = "display#quality"
    text_pair = "mk16i用后的体验感觉不错，就是有点厚，屏幕分辨率高，运行流畅，就是不知道能不能刷4.0的系统啊"
    predict(text, text_pair, model, tokenizer, id2label, max_seq_len=args.max_seq_len)
    
