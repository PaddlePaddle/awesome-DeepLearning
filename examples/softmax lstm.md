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

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting beautifulsoup4
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/41/e6495bd7d3781cee623ce23ea6ac73282a373088fcd0ddc809a047b18eae/beautifulsoup4-4.9.3-py3-none-any.whl (115kB)
    [K     |████████████████████████████████| 122kB 22.8MB/s eta 0:00:01
    [?25hCollecting soupsieve>1.2; python_version >= "3.0" (from beautifulsoup4)
      Downloading https://mirror.baidu.com/pypi/packages/36/69/d82d04022f02733bf9a72bc3b96332d360c0c5307096d76f6bb7489f7e57/soupsieve-2.2.1-py3-none-any.whl
    Installing collected packages: soupsieve, beautifulsoup4
    Successfully installed beautifulsoup4-4.9.3 soupsieve-2.2.1



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

## 层次softmax

 ![](https://ai-studio-static-online.cdn.bcebos.com/0f4d3be85aab43a6834eb36c0408c936de0b4040668c47218de9f1845f3aef0f)

  首先对所有在V词表的词，根据词频来构建赫夫曼tree，词频越大，路径越短，编码信息更少。tree中的所有的叶子节点构成了词 V，中间节点则共有V-1个，上面的每个叶子节点存在唯一的从根到该节点的path。
  上图假设我们需要计算w2的输出概率，我们定义从根节点开始，每次经过中间节点，做一个二分类任务（左边或者右边），所以我们定义中间节点的n左边概率为 ：

  ![](https://ai-studio-static-online.cdn.bcebos.com/0750c1d49cbd40d1b5dcd4e3426e5a938ab7aeb832904afd8956086623e3a3f7)

  那么右边概率为：

  ![](https://ai-studio-static-online.cdn.bcebos.com/a88d72ebd3f3478a9fcce462815dfc8421046dfd2f3943f9998dd8c52746c15c)

  从根节点到w2，我们可以计算概率值为：

 ![](https://ai-studio-static-online.cdn.bcebos.com/a2056b2962914486aefbac2a127114098d4bffb782b940778724d350530b9f53)

  所以每次预测所有叶子节点的概率之和为1，是一个分布，与softmax一致。

  不同softmax的是，每个词word对应的是一个V大小的one-hot label，hierarchical softmax中每个叶子节点word，对应的label是赫夫曼编码，一般长度不超过logV，在训练的时候，每个叶子节点的label统一编码到一个固定的长度，不足的可以进行pad。这样我们就将复杂度从o（V）降到了o（logV）。

## LSTM可实现其他类型的NLP任务

### 序列到类别——文本分类

* 准备数据集
* 将数据集中的所有字映射为字典，使得每个字都有唯一的标号对应
* 实现embedding层
* 添加LSTM层进行特征抽取

![](https://ai-studio-static-online.cdn.bcebos.com/b7812bdf1a8245e79e24ba32ea33ecf8014f906518fb435788fecb127a67e611)


### 同步的序列到序列——中英翻译

* 准备数据集
* 使用LSTM编码，由encoder得到整句话的embedding，将输入转化成了一个向量
* 将上述得到的向量放到decoder中解码，即将得到的embedding逐次解码映射到词典中的某个词，找出概率最高的词，然后作为输出

![](https://ai-studio-static-online.cdn.bcebos.com/050f6e03502048c8b09b22e2d2333c29e371532aae0a4d6983371ce2854855e3)


### 异步的序列到序列——古诗生成

* 数据处理，按照字的出现频率建立字符集词典，根据词典得到每个字对应的索引号，建立从字符到索引号和索引号到字符两个字典。
* 生成训练集
* 建立模型，使用两个LSTM叠加上一个全连接层再进行训练
* 输入文本，开始预测

![](https://ai-studio-static-online.cdn.bcebos.com/6338001e9913488c9a91bb51af6264a9086d2eb8432f4052a9df73edb45d8904)

