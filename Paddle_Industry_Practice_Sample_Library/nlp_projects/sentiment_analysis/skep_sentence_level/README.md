# 基于SKEP预训练模型进行情感分析

众所周知，人类自然语言中包含了丰富的情感色彩：表达人的情绪（如悲伤、快乐）、表达人的心情（如倦怠、忧郁）、表达人的喜好（如喜欢、讨厌）、表达人的个性特征和表达人的立场等等。情感分析在商品喜好、消费决策、舆情分析等场景中均有应用。利用机器自动分析这些情感倾向，不但有助于帮助企业了解消费者对其产品的感受，为产品改进提供依据；同时还有助于企业分析商业伙伴们的态度，以便更好地进行商业决策。

一般来讲，被人们所熟知的情感分析任务是语句级别的情感分析，其主要分析一段文本中整体蕴含的情感色彩。其常用于电影评论分析、网络论坛舆情分析等场景，如下面这句话所示。

> 15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错	1


为方便语句级别的情感分析任务建模，可将情感极性分为正向、负向、中性三个类别，这样将情感分析任务转变为一个分类问题，如图1所示：

- 正向： 表示正面积极的情感，如高兴，幸福，惊喜，期待等；
- 负向： 表示负面消极的情感，如难过，伤心，愤怒，惊恐等；
- 中性： 其他类型的情感；

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b630901b397e4e7a8e78ab1d306dfa1fc070d91015a64ef0b8d590aaa8cfde14" width="600" ></center>
<br><center>图1 情感分析任务</center></br>

本实践将基于预训练SKEP模型，在[ChnSentiCorp](https://aistudio.baidu.com/aistudio/competition/detail/50/0/task-definition)
 进行建模语句级别的情感分析任务。



## 1. 方案设计

本实践的设计思路如图2所示，首先将文本串传入SKEP模型中，利用SKEP模型对该文本串进行语义编码后，将CLS位置的token输出向量作为最终的语义编码向量。接下来将根据该语义编码向量进行情感分类。需要注意的是：[ChnSentiCorp](https://aistudio.baidu.com/aistudio/competition/detail/50/0/task-definition) 数据集是个二分类数据集，其情感极性只包含正向和负向两个类别。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/fc21e1201154451a80f32e0daa5fa84386c1b12e4b3244e387ae0b177c1dc963" /></center>
<br><center>图2 语句级情感分析建模图</center><br/>

## 2. 数据说明
本实践将在数据集[ChnSentiCorp](https://aistudio.baidu.com/aistudio/competition/detail/50/0/task-definition) 进行情感分析任务。该数据集是个语句级别的数据集，其中训练集包含9.6k训练样本，开发集和测试集均包含1.2k训练样本。下面展示了该数据集中的两条样本：

```
{
    "text":"不错，帮朋友选的，朋友比较满意，就是USB接口少了，而且全是在左边，图片有误",
    "label":1
}

{
    "text":"机器背面似乎被撕了张什么标签，残胶还在。但是又看不出是什么标签不见了，该有的都在，怪",
    "label":0
}

```


## 3. 使用说明
### 3.1 模型训练
使用如下命令，进行语句级情感分析模型训练。

```shell
sh run_train.sh
```

### 3.2 模型测试
使用如下命令，进行语句级情感分析模型测试。

```shell
sh run_train.sh
```

### 3.3 模型推理
使用如下命令，进行语句级情感分析推理。
```shell
sh run_predict.sh
```
