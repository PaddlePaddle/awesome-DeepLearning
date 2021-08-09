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

import paddle
from sklearn.metrics import classification_report

def compute_metrics(labels, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    labels=[x for batch in labels for x in batch]
    outputs = []
    nb_correct=0
    nb_true=0
    val_f1s=[]
    label_vals=[0,1,2,3]
    y_trues=[]
    y_preds=[]
    for idx, end in enumerate(lens):
        y_true = labels[idx][:end].tolist()
        y_pred = [x for x in decodes[idx][:end]]
        nb_correct += sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
        nb_true+=len(y_true)
        y_trues.extend(y_true)
        y_preds.extend(y_pred)

    score = nb_correct / nb_true
    # val_f1 = metrics.f1_score(y_trues, y_preds, average='micro', labels=label_vals)

    result=classification_report(y_trues, y_preds)
    # print(val_f1)   
    return score,result
    
def evaluate(model, loss_fct, data_loader, label_num):
    '''
    模型评估
    '''
    model.eval()
    pred_list = []
    len_list = []
    labels_list=[]
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(length.numpy())
        labels_list.append(labels.numpy())
    accuracy, result=compute_metrics(labels_list, pred_list, len_list)
    print("eval loss: %f, accuracy: %f" % (avg_loss, accuracy))
    print(result)
    model.train()

def write2txt(args, preds):
    '''
    将预测结果导入到txt文件
    '''
    file_path = args.output_pred_path
    with open(file_path, "w", encoding="utf8") as fout:
        fout.write("\n".join(preds))
        # Print some examples
    print("The results have been saved in the file: %s, 5 examples are shown below: " % file_path)
    print("\n".join(preds[:5]))   