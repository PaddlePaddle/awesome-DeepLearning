import paddle
import paddle.nn as nn

import paddlenlp
from functools import partial
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.embeddings import TokenEmbedding
from utils import load_dict, evaluate, predict, parse_decodes1, parse_decodes2
from paddlenlp.transformers import ErnieGramTokenizer, ErnieGramForTokenClassification
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
word_vocab = load_dict('./conf/word.dic')

# 将token转换为id
def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids

# 将文本和label转换为id
def convert_example(example):
        tokens, labels = example
        token_ids = convert_tokens_to_ids(tokens, word_vocab, 'OOV')
        label_ids = convert_tokens_to_ids(labels, label_vocab, 'O')
        return token_ids, len(token_ids), label_ids

# 调用内置的map()方法，进行数据处理：转换id
train_ds.map(convert_example)
dev_ds.map(convert_example)
test_ds.map(convert_example)

batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get('OOV')),  # token_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=label_vocab.get('O'))  # label_ids
    ): fn(samples)

train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=32,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=32,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

class BiGRUWithCRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 word_num,
                 label_num,
                 use_w2v_emb=False):
        super(BiGRUWithCRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(
                extended_vocab_path='./conf/word.dic', unknown_token='OOV')
        else:
            self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size,
                          hidden_size,
                          num_layers=2,
                          direction='bidirectional')
        self.fc = nn.Linear(hidden_size * 2, label_num + 2)  # BOS EOS
        self.crf = LinearChainCrf(label_num)
        self.decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, x, lens):
        embs = self.word_emb(x)
        output, _ = self.gru(embs)
        output = self.fc(output)
        _, pred = self.decoder(output, lens)
        return output, lens, pred

# Define the model netword and its loss
network = BiGRUWithCRF(300, 300, len(word_vocab), len(label_vocab))
model = paddle.Model(network)


optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
crf_loss = LinearChainCrfLoss(network.crf)
chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
model.prepare(optimizer, crf_loss, chunk_evaluator)


model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=10,
              save_dir='./gru_results',
              log_freq=1)
model.evaluate(eval_data=test_loader, log_freq=1)

outputs, lens, decodes = model.predict(test_data=test_loader)
preds = parse_decodes1(test_ds, decodes, lens, label_vocab)
print(len(preds))
print('\n'.join(preds[:5]))

