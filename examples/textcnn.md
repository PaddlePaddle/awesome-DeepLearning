###TextCNN 模型：

textcnn:将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。由Yoon Kim在论文(2014 EMNLP) Convolutional Neural Networks for Sentence Classification提出TextCNN。

网络结构：
https://img2018.cnblogs.com/blog/1182656/201809/1182656-20180919171652802-136261435.png

详细过程原理：
https://img2018.cnblogs.com/blog/1182656/201809/1182656-20180919171920103-1233770993.png
TextCNN详细过程：

Embedding：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点。
Convolution：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel。
MaxPolling：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。
FullConnection and Softmax：最后接一层全连接的 softmax 层，输出每个类别的概率。
  

通道（Channels）：

图像中可以利用 (R, G, B) 作为不同channel,文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。
 

一维卷积（conv-1d）：

图像是二维数据,文本是一维数据，因此在TextCNN卷积用的是一维卷积（在word-level上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要通过设计不同 kernel_size 的 filter 获取不同宽度的视野。

textcnn使用预先训练好的词向量作embedding layer。对于数据集里的所有词，因为每个词都可以表征成一个向量，因此我们可以得到一个嵌入矩阵MM, MM里的每一行都是词向量。这个MM可以是静态(static)的，也就是固定不变。可以是非静态(non-static)的，也就是可以根据反向传播更新。

多种模型：Convolutional Neural Networks for Sentence Classification文章中给出了几种模型，其实这里基本都是针对Embedding layer做的变化。

CNN-rand

        作为一个基础模型，Embedding layer所有words被随机初始化，然后模型整体进行训练。

CNN-static

        模型使用预训练的word2vec初始化Embedding layer，对于那些在预训练的word2vec没有的单词，随机初始化。然后固定Embedding layer，fine-tune整个网络。

CNN-non-static

        同（2），只是训练的时候，Embedding layer跟随整个网络一起训练。

CNN-multichannel

        Embedding layer有两个channel，一个channel为static，一个为non-static。然后整个网络fine-tune时只有一个channel更新参数。两个channel都是使用预训练的word2vec初始化的。
 

Pooling层：

利用CNN解决文本分类问题的文章还是很多的，比如这篇 A Convolutional Neural Network for Modelling Sentences 最有意思的输入是在 pooling 改成 (dynamic) k-max pooling ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息。

CNN中采用Max Pooling操作有几个好处：首先，这个操作可以保证特征的位置与旋转不变性，因为不论这个强特征在哪个位置出现，都会不考虑其出现位置而能把它提出来。但是对于NLP来说，这个特性其实并不一定是好事，因为在很多NLP的应用场合，特征的出现位置信息是很重要的，比如主语出现位置一般在句子头，宾语一般出现在句子尾等等。其次，MaxPooling能减少模型参数数量，有利于减少模型过拟合问题。因为经过Pooling操作后，往往把2D或者1D的数组转换为单一数值，这样对于后续的Convolution层或者全联接隐层来说无疑单个Filter的参数或者隐层神经元个数就减少了。再者，对于NLP任务来说，可以把变长的输入X整理成固定长度的输入。因为CNN最后往往会接全联接层，而其神经元个数是需要事先定好的，如果输入是不定长的那么很难设计网络结构。但是，CNN模型采取MaxPooling Over Time也有缺点：首先特征的位置信息在这一步骤完全丢失。在卷积层其实是保留了特征的位置信息的，但是通过取唯一的最大值，现在在Pooling层只知道这个最大值是多少，但是其出现位置信息并没有保留；另外一个明显的缺点是：有时候有些强特征会出现多次，出现次数越多说明这个特征越强，但是因为Max Pooling只保留一个最大值，就是说同一特征的强度信息丢失了。

###实验参数分析
	TextCNN模型中，超参数主要有词向量，Region Size的大小，Feature Map的数量，激活函数的选择，Pooling的方法，正则化的影响。《A Sensitivity Analysis...》论文前面几章对实验内容和结果进行了详细介绍，在9个数据集上基于Kim Y的模型做了大量的调参实验，得出AUC进行比较，根据的实验对比：

1）初始化词向量：一般不直接使用One-hot。除了随机初始化Embedding layer的外，使用预训练的word2vec、 GloVe初始化的效果都更加好（具体哪个更好依赖于任务本身）。非静态的比静态的效果好一些。

2）卷积核的尺寸filter_sizes：影响较大，通常过滤器的大小范围在1-10之间，一般取为3-5，对于句子较长的文本（100+），则应选择大一些。为了找到最优的过滤器大小(Filter Region Size)，可以使用线性搜索的方法。对不同尺寸ws的窗口进行结合会对结果产生影响。当把与最优ws相近的ws结合时会提升效果，但是如果将距离最优ws较远的ws相结合时会损害分类性能。刚开始，我们可以只用一个filter，调节Region Size来比对各自的效果，来看看那种size有最好的表现，然后在这个范围在调节不同Region的匹配。

3）卷积核的数量num_filters（对每个巻积核尺寸来说）：有较大的影响，一般取100~600（需要兼顾模型的训练效率） ，同时一般使用Dropout（0~0.5）。最好不要超过600，超过600可能会导致过拟合。可设为100-200。

4）激活函数：可以尽量多尝试激活函数，实验发现ReLU和tanh两种激活函数表现较佳。

5）池化选择：1-max pooling（1-max pooling的方式已经足够好了，相比于其他的pooling方式而言）。

6）Dropout和正则化：Dropout rate / dropout_keep_prob：dropout一般设为0.5。随着feature map数量增加，性能减少时，可以考虑增大正则化的力度，如尝试大于0.5的Dropout。正则化的作用微乎其微，正则项对最终模型性能的影响很小。l2正则化效益很小，所以这里建议设置一个比较大的L2 norm constrain，相比而言，dropout在神经网络中有着广泛的使用和很好的效果。

7）为了检验模型的性能水平，多次反复的交叉验证是必要的，这可以确保模型的高性能并不是偶然。

8） 随机性影响：由于模型训练过程中的随机性因素，如随机初始化的权重参数，mini-batch，随机梯度下降优化算法等，造成模型在数据集上的结果有一定的浮动，如准确率(accuracy)能达到1.5%的浮动，而AUC则有3.4%的浮动。



```python
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

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/keras/__init__.py in <module>
          2 try:
    ----> 3     from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
          4 except ImportError:


    ModuleNotFoundError: No module named 'tensorflow'

    
    During handling of the above exception, another exception occurred:


    ImportError                               Traceback (most recent call last)

    <ipython-input-15-684697d735c5> in <module>
          1 import logging
          2 
    ----> 3 from keras import Input
          4 from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
          5 from keras.models import Model


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/keras/__init__.py in <module>
          4 except ImportError:
          5     raise ImportError(
    ----> 6         'Keras requires TensorFlow 2.2 or higher. '
          7         'Install TensorFlow via `pip install tensorflow`')
          8 


    ImportError: Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`

