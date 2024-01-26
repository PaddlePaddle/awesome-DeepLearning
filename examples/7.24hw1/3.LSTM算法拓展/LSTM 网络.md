

# LSTM 网络

Long Short Term 网络—— 一般就叫做 LSTM ——是一种 RNN 特殊的类型，可以学习长期依赖信息。LSTM 由[Hochreiter & Schmidhuber (1997)](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)提出，并在近期被[Alex Graves](https://scholar.google.com/citations?user=DaFHynwAAAAJ&hl=en)进行了改良和推广。在很多问题，LSTM 都取得相当巨大的成功，并得到了广泛的使用。
LSTM 通过刻意的设计来避免长期依赖问题。记住长期的信息在实践中是 LSTM 的默认行为，而非需要付出很大代价才能获得的能力！
所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。

![img](https://img-blog.csdnimg.cn/img_convert/dbb59087899ca578f8aa37d6ddb92eed.png)

标准 RNN 中的重复模块包含单一的层


LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于 单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。

![img](https://img-blog.csdnimg.cn/img_convert/c62bac00b397ecdeff7da9dee3bec447.png)

LSTM 中的重复模块包含四个交互的层


不必担心这里的细节。我们会一步一步地剖析 LSTM 解析图。现在，我们先来熟悉一下图中使用的各种元素的图标。

![img](https://img-blog.csdnimg.cn/img_convert/93b66bae065c921ae08f50e71efc211f.png)

LSTM 中的图标


在上面的图例中，每一条黑线传输着一整个向量，从一个节点的输出到其他节点的输入。粉色的圈代表 pointwise 的操作，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。合在一起的线表示向量的连接，分开的线表示内容被复制，然后分发到不同的位置。

# LSTM 的核心思想

LSTM 的关键就是细胞状态，水平线在图上方贯穿运行。
细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性 交互。信息在上面流传保持不变会很容易。

![img](https://img-blog.csdnimg.cn/img_convert/03b1addcef666d6bc2021c784d7ab064.png)

Paste_Image.png

LSTM 有通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。

![img](https://img-blog.csdnimg.cn/img_convert/953aa4680e08ae0619adf4668f9f9397.png)

Paste_Image.png


Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”！

LSTM 拥有三个门，来保护和控制细胞状态。

# 逐步理解 LSTM

在我们 LSTM 中的第一步是决定我们会从细胞状态中丢弃什么信息。这个决定通过一个称为**忘记门层**完成。该门会读取 `h_{t-1}` 和 `x_t`，输出一个在 0 到 1 之间的数值给每个在细胞状态 `C_{t-1}` 中的数字。1 表示“完全保留”，0 表示“完全舍弃”。
让我们回到语言模型的例子中来基于已经看到的预测下一个词。在这个问题中，细胞状态可能包含当前**主语**的性别，因此正确的**代词**可以被选择出来。当我们看到新的**主语**，我们希望忘记旧的**主语**。

![img](https://img-blog.csdnimg.cn/img_convert/5aa338be8a2d5b82725aac5cacc314cd.png)

决定丢弃信息


下一步是确定什么样的新信息被存放在细胞状态中。这里包含两个部分。第一，sigmoid 层称 “输入门层” 决定什么值我们将要更新。然后，一个 tanh 层创建一个新的候选值向量，`\tilde{C}_t`，会被加入到状态中。下一步，我们会讲这两个信息来产生对状态的更新。
在我们语言模型的例子中，我们希望增加新的主语的性别到细胞状态中，来替代旧的需要忘记的主语。

![img](https://img-blog.csdnimg.cn/img_convert/2a11557cabdf88bc3952008d3baa48c6.png)

确定更新的信息

现在是更新旧细胞状态的时间了，`C_{t-1}` 更新为 `C_t`。前面的步骤已经决定了将会做什么，我们现在就是实际去完成。
我们把旧状态与 `f_t` 相乘，丢弃掉我们确定需要丢弃的信息。接着加上 `i_t * \tilde{C}_t`。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。
在语言模型的例子中，这就是我们实际根据前面确定的目标，丢弃旧代词的性别信息并添加新的信息的地方。

![img](https://img-blog.csdnimg.cn/img_convert/2f4974718e5d89af051448ef59ff5161.png)

更新细胞状态

最终，我们需要确定输出什么值。这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。首先，我们运行一个 sigmoid 层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在 -1 到 1 之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。
在语言模型的例子中，因为他就看到了一个 **代词**，可能需要输出与一个 **动词** 相关的信息。例如，可能输出是否代词是单数还是负数，这样如果是动词的话，我们也知道动词需要进行的词形变化。

![img](https://img-blog.csdnimg.cn/img_convert/25a8126ed5b26f6d8699d822e9f8f3f7.png)

输出信息

# LSTM 的变体

我们到目前为止都还在介绍正常的 LSTM。但是不是所有的 LSTM 都长成一个样子的。实际上，几乎所有包含 LSTM 的论文都采用了微小的变体。差异非常小，但是也值得拿出来讲一下。
其中一个流形的 LSTM 变体，就是由 [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf) 提出的，增加了 “peephole connection”。是说，我们让 门层 也会接受细胞状态的输入。

![img](https://img-blog.csdnimg.cn/img_convert/6dd8683b944ab0b33f9ac7bed8a48a09.png)

peephole 连接

上面的图例中，我们增加了 peephole 到每个门上，但是许多论文会加入部分的 peephole 而非所有都加。

另一个变体是通过使用 coupled 忘记和输入门。不同于之前是分开确定什么忘记和需要添加什么新的信息，这里是一同做出决定。我们仅仅会当我们将要输入在当前位置时忘记。我们仅仅输入新的值到那些我们已经忘记旧的信息的那些状态 。

![img](https://img-blog.csdnimg.cn/img_convert/068e9013d692b55970668501540f3ef2.png)

coupled 忘记门和输入门


另一个改动较大的变体是 Gated Recurrent Unit (GRU)，这是由 [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf) 提出。它将忘记门和输入门合成了一个单一的 更新门。同样还混合了细胞状态和隐藏状态，和其他一些改动。最终的模型比标准的 LSTM 模型要简单，也是非常流行的变体。

![img](https://img-blog.csdnimg.cn/img_convert/1653c8c07d22805f7ad617f608a3ead3.png)

GRU


这里只是部分流行的 LSTM 变体。当然还有很多其他的，如[Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf) 提出的 Depth Gated RNN。还有用一些完全不同的观点来解决长期依赖的问题，如[Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf) 提出的 Clockwork RNN。
要问哪个变体是最好的？其中的差异性真的重要吗？[Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf) 给出了流行变体的比较，结论是他们基本上是一样的。[Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) 则在超过 1 万种 RNN 架构上进行了测试，发现一些架构在某些任务上也取得了比 LSTM 更好的结果。

![img](https://img-blog.csdnimg.cn/img_convert/4e6c07f8ce01d2ed65d7ad4eca706f8d.png)

