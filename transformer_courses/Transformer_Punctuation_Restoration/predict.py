
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

def parse_decodes(input_words, id2label, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        sent = input_words[idx]['tokens']
        tags = [id2label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        for s, t in zip(sent, tags):
            if(t=='0'):
                sent_out.append(s)
            else:
                # sent_out.append(s)
                sent_out.append(s+punctuation_dec[t])
        sent=' '.join(sent_out)
        sent=sent.replace(' ##','')
        outputs.append(sent)
    return outputs

if __name__ == '__main__':
     
    # 加载训练好的模型
    best_model = 'model_13257.pdparams'
    init_checkpoint_path=os.path.join(output_dir,'model_13257.pdparams')
    model_dict = paddle.load(init_checkpoint_path)
    model.set_dict(model_dict)
     
    punctuation_dec = {
            '0': 'O',
            '1': ',',
            '2': '.',
            '3': '?',
        }
 
    id2label = dict(enumerate(test_ds.label_list))
    raw_data = test_ds.data
 
    model.eval()
    pred_list = []
    len_list = []
    for step, batch in enumerate(test_data_loader):
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(length.numpy())
    preds = parse_decodes(raw_data, id2label, pred_list, len_list)

    # 写入到文件
    file_path = "results.txt"
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
        # Print some examples
    print("The results have been saved in the file: %s, some examples are shown below: " % file_path)
    print("\n".join(preds[:5]))
