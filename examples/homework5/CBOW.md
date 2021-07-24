```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```

**CBOW的算法实现**
对比Skip-gram，CBOW和Skip-gram的算法实现如 图1 所示。本项目将补充CBOW的算法实现过程
![](https://ai-studio-static-online.cdn.bcebos.com/891c477874fa48f4b51bb01a13a7bf282adca12725e74935a46a0e0275c128a2)

图1：CBOW和Skip-gram的算法实现


如 图1 所示，CBOW是一个具有3层结构的神经网络，分别是：

Input Layer（输入层）：接收one-hot张量 V∈R1×vocab_size作为网络的输入，里面存储着当前句子中上下文单词的one-hot表示。
Hidden Layer（隐藏层）：将张量V乘以一个word embedding张量W1∈Rvocab_size×embed_size ，并把结果作为隐藏层的输出，得到一个形状为R1×embed_size的张量，里面存储着当前句子上下文的词向量。
Output Layer（输出层）：将隐藏层的结果乘以另一个word embedding张量W2∈Rembed_size×vocab_size，得到一个形状为R1×vocab_size的张量。这个张量经过softmax变换后，就得到了使用当前上下文对中心的预测结果。根据这个softmax的结果，我们就可以去训练词向量模型。

在实际操作中，使用一个滑动窗口（一般情况下，长度是奇数），从左到右开始扫描当前句子。每个扫描出来的片段被当成一个小句子，每个小句子中间的词被认为是中心词，其余的词被认为是这个中心词的上下文。

CBOW算法和skip-gram算法最本质的区别就是：CBOW算法是以上下文预测中心词，而skip-gram算法是以中心城预测上下文。

**CBOW的理想实现**
使用神经网络实现CBOW中，模型接收的输入应该有2个不同的tensor：

代表当前上下文的tensor：假设我们称之为context_words V，一般来说，这个tensor是一个形状为[batch_size, vocab_size]的one-hot tensor，表示在一个mini-batch中，每组上下文中每一个单词的ID。

代表目标词的tensor：假设我们称之为target_words T，一般来说，这个tensor是一个形状为[batch_size, 1]的整型tensor，这个tensor中的每个元素是一个[0, vocab_size-1]的值，代表目标词的ID。

在理想情况下，我们可以这样实现CBOW：把上下文中的每一个单词，依次作为输入，把当前句子中的中心词作为标签，构建神经网络进行学习，实现上下文预测中心词。具体过程如下：

声明一个形状为[vocab_size, embedding_size]的张量，作为需要学习的词向量，记为W0对于给定的输入V，即某一个上下文的单词，使用向量乘法，将V乘以W0,这样就得到了一个形状为[batch_size, embedding_size]的张量，记为H=V∗W0。这个张量H就可以看成是经过词向量查表后的结果。

声明另外一个需要学习的参数W1 ，这个参数的形状为[embedding_size, vocab_size]。将上一步得到的H去乘以W1 ，得到一个新的tensor O=H∗W1，此时的OOO是一个形状为[batch_size, vocab_size]的tensor，表示当前这个mini-batch中的每一组上下文中的每一个单词预测出的目标词的概率。

使用softmax函数对mini-batch中每个中心词的预测结果做归一化，即可完成网络构建。

**CBOW的实际实现**
和课程中讲解的skip-gram一样，在实际中，为避免过于庞大的计算量，我们通常采用负采样的方法，来避免查询整个此表，从而将多分类问题转换为二分类问题。具体实现过程如图2：
![](https://ai-studio-static-online.cdn.bcebos.com/b5989d7667554733ac2ce18561e885c7cf49d23a199349f9ab8187aeaa08603f)
图2 CBOW算法的实际实现


在实现的过程中，通常会让模型接收3个tensor输入：

代表上下文单词的tensor：假设我们称之为context_words V，一般来说，这个tensor是一个形状为[batch_size, vocab_size]的one-hot tensor，表示在一个mini-batch中每个中心词具体的ID。

代表目标词的tensor：假设我们称之为target_words T，一般来说，这个tensor同样是一个形状为[batch_size, vocab_size]的one-hot tensor，表示在一个mini-batch中每个目标词具体的ID。

代表目标词标签的tensor：假设我们称之为labels L，一般来说，这个tensor是一个形状为[batch_size, 1]的tensor，每个元素不是0就是1（0：负样本，1：正样本）。

模型训练过程如下：

首先遍历上下文，得到上下文中的一个单词，用VVV（上下文）去查询W0，用T（目标词）去查询W1 ，分别得到两个形状为[batch_size, embedding_size]的tensor，记为H1和H2 。
点乘这两个tensor，最终得到一个形状为[batch_size]的tensor O=[Oi=∑jH0[i,j]∗H1[i,j]]i=1batch_size。

使用随即负采样得到一些负样本（0），同时以目标词作为正样本（1），输入值标签信息label。

使用sigmoid函数作用在O上，将上述点乘的结果归一化为一个0-1的概率值，作为预测概率，根据标签信息label训练这个模型即可。



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
