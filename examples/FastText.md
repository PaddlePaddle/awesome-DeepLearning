```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

**fastText简介**

FastText是一个快速文本分类算法，是facebook开源的一个词向量与文本分类工具，在2016年开源，典型应用场景是“带监督的文本分类问题”。提供简单而高效的文本分类和表征学习的方法，性能比肩深度学习而且速度更快。

FastText结合了自然语言处理和机器学习中最成功的理念。这些包括了使用词袋以及n-gram袋表征语句，还有使用子词(subword)信息，并通过隐藏表征在类别间共享信息,另外采用了一个Softmax层级(利用了类别不均衡分布的优势)来加速运算过程。

FastText能够学会“男孩”、“女孩”、“男人”、“女人”指代的是特定的性别，并且能够将这些数值存在相关文档中。当某个程序在提出一个用户请求（假设是“我女友现在在儿？”），它能够马上在fastText生成的文档中进行查找并且理解用户想要问的是有关女性的问题。

与基于神经网络的分类算法相比,FastText有两大优点：

1、FastText在保持高精度的情况下加快了训练速度和测试速度

2、FastText不需要预训练好的词向量，FastText会自己训练词向量

3、FastText两个重要的优化：Hierarchical Softmax、N-gram

**FastText原理**

fastText方法包含三部分，模型架构，层次SoftMax和N-gram特征。

**模型架构**

fastText的架构和word2vec中的CBOW的架构类似，因为它们的作者都是Facebook的科学家Tomas Mikolov，而且确实fastText也算是words2vec所衍生出来的。

CBOW的架构:输入的是w(t)的上下文2d个词，经过隐藏层后，输出的是w(t)

![](https://ai-studio-static-online.cdn.bcebos.com/572a37232b5047549ab6ec28cd561928df464e70870841598e7742f9b0245fd2)

ord2vec将上下文关系转化为多分类任务，进而训练逻辑回归模型，这里的类别数量是 |V| 词库大小。通常的文本数据中，词库少则数万，多则百万，在训练中直接训练多分类逻辑回归并不现实。

word2vec中提供了两种针对大规模多分类问题的优化手段， negative sampling 和 hierarchical softmax。在优化中，negative sampling 只更新少量负面类，从而减轻了计算量。

hierarchical softmax 将词库表示成前缀树，从树根到叶子的路径可以表示为一系列二分类器，一次多分类计算的复杂度从|V|降低到了树的高度。

**fastText模型架构:**

其中x1,x2,...,xN−1,xN表示一个文本中的n-gram向量，每个特征是词向量的平均值。这和前文中提到的cbow相似，cbow用上下文去预测中心词，而此处用全部的n-gram去预测指定类别。

![](https://ai-studio-static-online.cdn.bcebos.com/7c85f97a770c4723bd65f74ee37dfdf379e36dc2e80344ffa3898877eacea6b9)

其中x1,x2,...,xN−1,xNx1,x2,...,xN−1,xN表示一个文本中的n-gram向量，每个特征是词向量的平均值。什么是n-gram呢？

如果n为3，也叫作trigram，最小切分单位为字，则“欢迎关注数据科学杂谈”这个句子的3-gram为{“欢迎关”，“迎关注”，“关注数”,“注数据”，“数据科”，“据科学”，“科学杂”，“学杂谈”}。n-gram除了获取上下文信息，还能将语言的局部顺序保持住，想想看“羊吃草”，如果不考虑顺序，可能会有“草吃羊”的语义不正确问题。


**层次SoftMax**

对于有大量类别的数据集，fastText使用了一个分层分类器（而非扁平式架构）。不同的类别被整合进树形结构中（想象下二叉树而非 list）。在某些文本分类任务中类别很多，计算线性分类器的复杂度高。为了改善运行时间，fastText 模型使用了层次 Softmax 技巧。层次 Softmax 技巧建立在哈弗曼编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量。

fastText 也利用了类别（class）不均衡这个事实（一些类别出现次数比其他的更多），通过使用 Huffman 算法建立用于表征类别的树形结构。因此，频繁出现类别的树形结构的深度要比不频繁出现类别的树形结构的深度要小，这也使得进一步的计算效率更高。

因为每个字有字向量，例如之前CBOW得到的每个词的词向量，当我们使用n-gram时，我们将n个字向量取平均，例如我们将“欢”、“迎”、“关”三个字的向量平均后得到“欢迎关”这个词的向量。然后在隐藏层将得到的所有n-gram的词向量求平均，得到最终的一个向量。此时的情况有点像CBOW中得到的向量，和CBOW相似，此时的向量要经过一个softmax层，只是不同于CBOW的普通softmax，Fasttext中使用分层Softmax分类。分层Softmax分类作为其输出层。

在标准的softmax中，计算一个类别的softmax概率时，我们需要对所有类别概率做归一化，在这类别很大情况下非常耗时，因此提出了分层softmax，思想是根据类别的频率构造哈夫曼树来代替标准softmax，通过分层softmax可以将复杂度从N降低到logN，下图给出分层softmax示例：
![](https://ai-studio-static-online.cdn.bcebos.com/96d9bd3c17be4b2392f952697cccfa010f5d25157b0848e09f3b394c1d789eb5)


在层次softmax模型中，叶子结点的词没有直接输出的向量，而非叶子节点都有响应的输在在模型的训练过程中，通过Huffman编码，构造了一颗庞大的Huffman树，同时会给非叶子结点赋予向量。我们要计算的是目标词w的概率，这个概率的具体含义，是指从root结点开始随机走，走到目标词w的概率。因此在途中路过非叶子结点（包括root）时，需要分别知道往左走和往右走的概率。哈夫曼树中的每个叶子节点代表一种类别，在每一个非叶子节点处都要做一次二分类，走左边子树的概率为p，走右边子树的概率为1-p，这里的二分类都可以用逻辑回归表示。每一种分类都会有一条路径，也就是说，每个中间节点都是一个逻辑回归二分类器，而每个类别的概率为中间若干逻辑回归单元输出概率的连乘积。

**N-gram特征**

常用的特征是词袋模型（将输入数据转化为对应的Bow形式）。但词袋模型不能考虑词之间的顺序，因此 fastText 还加入了 N-gram 特征。n-gram是基于语言模型的算法，基本思想是将文本内容按照子节顺序进行大小为N的窗口滑动操作，最终形成窗口为N的字节片段序列。而且需要额外注意一点是n-gram可以根据粒度不同有不同的含义，有字粒度的n-gram和词粒度的n-gram。

“我 爱 她” 这句话中的词袋模型特征是 “我”，“爱”, “她”。这些特征和句子 “她 爱 我” 的特征是一样的。如果加入 2-Ngram，第一句话的特征还有 “我-爱” 和 “爱-她”，这两句话 “我 爱 她” 和 “她 爱 我” 就能区别开来了。当然，为了提高效率，我们需要过滤掉低频的 N-gram。

同时n-gram也可以在字符级别工作，例如对单个单词matter来说，假设采用3-gram特征，那么matter可以表示成图中五个3-gram特征，这五个特征都有各自的词向量，五个特征的词向量和即为matter这个词的向其中“<”和“>”是作为边界符号被添加，来将一个单词的ngrams与单词本身区分开来：

![](https://ai-studio-static-online.cdn.bcebos.com/50eb67a325034f4ba91538f8ff988813bc6aa28b63ef4733a7e08b7cecdf9714)

使用n-gram有如下优点
1、为罕见的单词生成更好的单词向量：根据上面的字符级别的n-gram来说，即是这个单词出现的次数很少，但是组成单词的字符和其他单词有共享的部分，因此这一点可以优化生成的单词向量
2、在词汇单词中，即使单词没有出现在训练语料库中，仍然可以从字符级n-gram中构造单词的词向量
3、n-gram可以让模型学习到局部单词顺序的部分信息, 如果不考虑n-gram则便是取每个单词，这样无法考虑到词序所包含的信息，即也可理解为上下文信息，因此通过n-gram的方式关联相邻的几个词，这样会让模型在训练的时候保持词序信息


**fastText代码实现**


```python
pip install fasttext
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting fasttext
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/f8/85/e2b368ab6d3528827b147fdb814f8189acc981a4bc2f99ab894650e05c40/fasttext-0.9.2.tar.gz (68kB)
    [K     |████████████████████████████████| 71kB 9.4MB/s  eta 0:00:01
    [?25hRequirement already satisfied: pybind11>=2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from fasttext) (2.7.0)
    Requirement already satisfied: setuptools>=0.7.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from fasttext) (41.4.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from fasttext) (1.16.4)
    Building wheels for collected packages: fasttext
      Building wheel for fasttext (setup.py) ... [?25ldone
    [?25h  Created wheel for fasttext: filename=fasttext-0.9.2-cp37-cp37m-linux_x86_64.whl size=3264287 sha256=e511248f2f5eea0349c5e773c4d3e043bf040d43c6c3fc83846067f819954e45
      Stored in directory: /home/aistudio/.cache/pip/wheels/9a/38/71/f555993ec7d3561a2f0a07a7d8c12ab2da6dadc08968ed48a1
    Successfully built fasttext
    Installing collected packages: fasttext
    Successfully installed fasttext-0.9.2
    Note: you may need to restart the kernel to use updated packages.



```python
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import pandas as pd
```

设计fasttext的代码结构。先将词转换为向量形式，然后将这些向量加起来求平均,再去分类。



```python
class FastText(nn.Module):
    def __init__(self, vocab, w2v_dim, classes, hidden_size):
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(len(vocab), w2v_dim)  #embedding初始化，需要两个参数，词典大小、词向量维度大小
        self.embed.weight.requires_grad = True #需要计算梯度，即embedding层需要被训练
        self.fc = nn.Sequential(              #序列函数
            nn.Linear(w2v_dim, hidden_size),  #这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(hidden_size),      #再进入一个BatchNorm1d
            nn.ReLU(inplace=True),            #再经过Relu激活函数
            nn.Linear(hidden_size, classes)#最后再经过一个线性变换
        )
    def forward(self, x):                      
        x = self.embed(x)                     #先将词id转换为对应的词向量
        out = self.fc(torch.mean(x, dim=1))   #这使用torch.mean()将向量进行平均
        return out
```

训练模型，主要要将网络的模式设置为训练模式




```python
def train_model(net, epoch, lr, data, label):      #训练模型
    print("begin training")
    net.train()  # 将模型设置为训练模式，很重要！
    optimizer = optim.Adam(net.parameters(), lr=lr) #设置优化函数
    Loss = nn.CrossEntropyLoss()  #设置损失函数
    for i in range(epoch):  # 循环
        optimizer.zero_grad()  # 清除所有优化的梯度
        output = net(data)  # 传入数据，前向传播，得到预测结果
        loss = Loss(output, target) #计算预测值和真实值之间的差异，得到loss
        loss.backward() #loss反向传播
        optimizer.step() #优化器优化参数

        # 打印状态信息
        print("train epoch=" + str(i) + ",loss=" + str(loss.item()))
    print('Finished Training')
```


```python
#验证代码，检测模型训练效果。同时，记得将网络模式调整为验证模式
def model_test(net, test_data, test_label):
    net.eval()  # 将模型设置为验证模式
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = net(test_data)
        # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
        _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
        total += test_label.size(0)
        correct += (predicted == test_label).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))
```


```python
if __name__ == "__main__":
    #这里没有写具体数据的处理方法，毕竟大家所做的任务不一样
    batch_size = 64
    epoch = 10  # 迭代次数
    w2v_dim = 300  # 词向量维度
    lr = 0.001
    hidden_size = 128
    classes = 2

    # 定义模型
    net = FastText(vocab=vocab, w2v_dim=w2v_dim, classes=classes, hidden_size=hidden_size)

    # 训练
    print("开始训练模型")
    train_model(net, epoch, lr, data, label)
    # 保存模型
    print("开始测试模型")
    model_test(net, test_data, test_label)
```


