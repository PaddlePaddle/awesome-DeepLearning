import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras


input_filepath = './data/train/tiny-shakespeare-train.txt'
text = open(input_filepath, 'r').read()
# 利用set方法取出字符，并将重复字符去掉，sort方法用来排序
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)
# 将text变为对应的idx组成的文本
text_as_int = np.array([char2idx[c] for c in text])


def split_input_target(id_text):
    """
    文本为abcde,则输入为abcd,四个字符对应的输出分别为：bcde,即每个输出都是输入的下一个字符
    """
    return id_text[0:-1], id_text[1:]


# 定义dataset
# 每个字符集对应的idx的dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# 定义一个句子型的dataset
seg_length = 100
seq_dataset = char_dataset.batch(seg_length + 1, drop_remainder=True)

# 设定batch_size
seq_dataset = seq_dataset.map(split_input_target)
batch_size = 64
buffer_size = 10000
seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(units=rnn_units,
                          stateful=True,
                          recurrent_initializer="glorot_uniform",
                          return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units,
                    batch_size=batch_size)
model.summary()
for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_prediction = model(input_example_batch)
    print(example_batch_prediction.shape)


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# 保存模型
output_dir = "./text_generation_lstm_checkpoints"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix,
                save_weights_only=True)

# 训练
# epochs = 10
# history = model.fit(seq_dataset, epochs=epochs,
#                     callbacks=[checkpoint_callback])


model2 = build_model(vocab_size,
                     embedding_dim,
                     rnn_units,
                     batch_size=1)
model2.load_weights(tf.train.latest_checkpoint(output_dir))

# 定义model2的输入shape
model2.build(tf.TensorShape([1, None]))
model2.summary()


temperature = 0.5


def generate_text(model, start_string, num_generate=1000):
    input_eval = [char2idx[ch] for ch in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions / temperature
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        text_generated.append(idx2char[predicted_id])
        input_eval = tf.expand_dims([predicted_id], 0)
    return start_string + ''.join(text_generated)


# 调用
new_text = generate_text(model2, "first: ")
print(new_text)