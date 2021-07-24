"""
构建一个简单的文本CNN网络，基于paddle动态图。测试
"""

from paddle import fluid


class CNN_pd(fluid.dygraph.Layer):
    """
    A cnn net for text classification, the framework is embedding - conv1d - pool - fc - out
    """
    def __init__(self, output_dim,
                 kernel_size=5,
                 dimension=100,
                 conv_filters=40,
                 stride=2,
                 act='relu',
                 words_num=10000,
                 use_bias=True,
                 padding_id=0):
        super(CNN_pd, self).__init__()
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dimension = dimension
        self.conv_filters = conv_filters
        self.stride = stride
        self.act = act
        self.words_num = words_num
        self.use_bias = use_bias
        self.padding_id = padding_id
        self.if_built = False
        self.embedding = fluid.Embedding(size=[self.words_num, self.dimension], is_sparse=True, padding_idx=self.padding_id,
                                         param_attr=fluid.ParamAttr(
                                             name='embedding',
                                             initializer=fluid.initializer.UniformInitializer(
                                                 low=-0.05, high=0.05)))
        self.conv1d = fluid.Conv2D(num_filters=self.conv_filters, stride=(self.stride, 1), num_channels=1,
                                   filter_size=(self.kernel_size, self.dimension), act=self.act, bias_attr=self.use_bias)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = fluid.layers.reshape(x, shape=[x.shape[0], 1, x.shape[1], x.shape[2]])
        x = self.conv1d(x)
        x = fluid.layers.pool2d(x, pool_size=(2, 1))
        x = fluid.layers.flatten(x)
        if self.if_built == False:
            self.fc = fluid.Linear(input_dim=x.shape[1], output_dim=self.output_dim, act='softmax')
            self.if_built = True
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import data_utils
    import numpy as np

    x, y, vocabulary, vocabulary_inv, test_size, _, _ = data_utils.load_data()
    train_x = x[:-test_size]
    train_y = y[:-test_size]
    test_x = x[-test_size:]
    test_y = y[-test_size:]

    def build_data(x, y, batch_size=64):
        len_x = x.shape[0]
        # shuffle
        shuffle_ix = np.random.permutation(np.arange(len_x))
        x = x[shuffle_ix]
        y = y[shuffle_ix]
        num = int(np.ceil(len_x / batch_size))
        for i in range(num):
            s = i * batch_size
            e = min(len_x, (i + 1) * batch_size)
            ix = np.arange(s, e)
            x_batch = x[ix]
            y_batch = y[ix]
            yield x_batch, y_batch, i+1, num

    epochs = 10
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        cnnpd = CNN_pd(output_dim=train_y[0].shape[0], kernel_size=5, dimension=100, conv_filters=40, stride=2,
                       act='relu',
                       words_num=len(vocabulary), use_bias=True)
        cnnpd.train()
        adam = fluid.optimizer.Adam(learning_rate=1e-3, parameter_list=cnnpd.parameters())

        history = {'acc': [], 'loss': []}
        for epoch in range(epochs):
            data = build_data(np.array(train_x).astype('int64'), np.array(train_y).astype('float32'), batch_size=32)
            ave_loss = 0
            for x_, y_, step, total_step in data:
                x_t = fluid.dygraph.to_variable(x_)
                y_t = fluid.dygraph.to_variable(y_)
                x_t.stop_gradient = False
                y_t.stop_gradient = False
                # forward
                pred = cnnpd(x_t)
                y_t = fluid.layers.argmax(y_t, axis=1)
                #print(y_t)
                loss = fluid.layers.cross_entropy(pred, y_t)
                #print(loss)
                loss = fluid.layers.reduce_mean(loss, dim=0)
                #loss = fluid.layers.reduce_sum(loss)
                # backward
                loss.backward()
                adam.minimize(loss)
                #print(loss.gradient())
                # clear gradient
                cnnpd.clear_gradients()
                ave_loss = (ave_loss * (float(step) - 1.0) + loss.numpy()[0]) / float(step)
                if step % 10 == 0 and step < total_step:
                    print('epoch: {} - step: {} / {} - loss: {:.4f}'.format(epoch + 1,
                                                                          str(step).ljust(
                                                                              len(str(total_step))),
                                                                          total_step,
                                                                          loss.numpy()[0]))
                if step == total_step:
                    print(
                        'epoch: {} done - ave loss: {:.4f}'.format(epoch + 1,  ave_loss))
            history['loss'].append(loss.numpy()[0])

    # 绘图
    import matplotlib.pyplot as plt

    epochs = [i + 1 for i in range(epochs)]
    plt.plot(epochs, history['loss'], 'r-*')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss trend')
    plt.show()


