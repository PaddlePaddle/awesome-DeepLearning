import logging

from keras import Input
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
from keras.models import Model
from keras.utils import plot_model


def textcnn(max_sequence_length, max_token_num, embedding_dim, output_dim, model_img_path=None, embedding_matrix=None):
    """ TextCNN: 1. embedding layers, 2.convolution layer, 3.max-pooling, 4.softmax layer. """
    x_input = Input(shape=(max_sequence_length,))
    logging.info("x_input.shape: %s" % str(x_input.shape))  # (?, 60)

    if embedding_matrix is None:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length)(x_input)
    else:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                          weights=[embedding_matrix], trainable=True)(x_input)
    logging.info("x_emb.shape: %s" % str(x_emb.shape))  # (?, 60, 300)

    pool_output = []
    kernel_sizes = [2, 3, 4] 
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=2, kernel_size=kernel_size, strides=1)(x_emb)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        logging.info("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))
    pool_output = concatenate([p for p in pool_output])
    logging.info("pool_output.shape: %s" % str(pool_output.shape))  # (?, 1, 6)

    x_flatten = Flatten()(pool_output)  # (?, 6)
    y = Dense(output_dim, activation='softmax')(x_flatten)  # (?, 2)
    logging.info("y.shape: %s \n" % str(y.shape))

    model = Model([x_input], outputs=[y])
    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    model.summary()
    return model

