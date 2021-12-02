import paddle
import paddle.nn as nn

import paddlenlp
from functools import partial
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
from utils import load_dict, evaluate, predict, parse_decodes1, parse_decodes2
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, ErnieGramTokenizer, ErnieGramForTokenClassification
from utils import convert_example

def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

train_ds, dev_ds, test_ds = load_dataset(datafiles=(
        './waybill_data/train.txt', './waybill_data/dev.txt', './waybill_data/test.txt'))


label_vocab = load_dict('./conf/tag.dic')

# 设置想要使用模型的名称
MODEL_NAME = "ernie-1.0"
tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)
test_ds.map(trans_func)
ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)

train_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_size=200,
    return_list=True,
    collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_size=200,
    return_list=True,
    collate_fn=batchify_fn)
test_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_size=200,
    return_list=True,
    collate_fn=batchify_fn)


# Define the model netword and its loss
model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))

metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

step = 0
for epoch in range(10):
    # Switch the model to training mode
    model.train()
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step += 1
        print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    evaluate(model, metric, dev_loader)

    paddle.save(model.state_dict(),
                './ernie_result/model_%d.pdparams' % step)
# model.save_pretrained('./checkpoint')
# tokenizer.save_pretrained('./checkpoint')

preds = predict(model, test_loader, test_ds, label_vocab)
file_path = "ernie_results.txt"
with open(file_path, "w", encoding="utf8") as fout:
    fout.write("\n".join(preds))
# Print some examples
print(
    "The results have been saved in the file: %s, some examples are shown below: "
    % file_path)
print("\n".join(preds[:10]))




