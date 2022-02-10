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

def convert_example_to_feature(example, tokenizer, max_seq_len=512):
    encoded_inputs = tokenizer(example["text"], max_seq_len=max_seq_len)
    labels = np.array(example["label"], dtype="int64")

    return encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], labels