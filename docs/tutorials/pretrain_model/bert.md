# BERT

## BERT介绍

+ BERT(Bidirectional Encoder Representation from Transformers)是2018年10月由Google AI研究院提出的一种预训练模型，该模型在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩: 全部两个衡量指标上全面超越人类，并且在11种不同NLP测试中创出SOTA表现，包括将GLUE基准推高至80.4% (绝对改进7.6%)，MultiNLI准确度达到86.7% (绝对改进5.6%)，成为NLP发展史上的里程碑式的模型成就。

+ BERT的网络架构使用的是《Attention is all you need》中提出的多层Transformer结构，如 **图1** 所示。其最大的特点是抛弃了传统的RNN和CNN，通过Attention机制将任意位置的两个单词的距离转换成1，有效的解决了NLP中棘手的长期依赖问题。Transformer的结构在NLP领域中已经得到了广泛应用。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/7b5e70561695477ea0c1b36f8ed6cbde577000b89d7748b99af4eeec1d1ab83a" width = "700"/> <br />
</p><br><center>图1 BERT结构</center></br>

## BERT的预训练任务

BERT是一个多任务模型，它的预训练（Pre-training）任务是由两个自监督任务组成，即MLM和NSP，如 **图2** 所示。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/dd2f05f637904f60b6e00f1d00ed94ad4d2199dcbe6e4be9826393ba3ac917b5" width = "700"/> <br />
</p><br><center>图2 BERT 预训练过程示意图</center></br>

+ MLM是指在训练的时候随即从输入预料上mask掉一些单词，然后通过的上下文预测该单词，该任务非常像我们在中学时期经常做的完形填空。正如传统的语言模型算法和RNN匹配那样，MLM的这个性质和Transformer的结构是非常匹配的。在BERT的实验中，15%的WordPiece Token会被随机Mask掉。在训练模型时，一个句子会被多次喂到模型中用于参数学习，但是Google并没有在每次都mask掉这些单词，而是在确定要Mask掉的单词之后，做以下处理。
	+ 80%的时候会直接替换为[Mask]，将句子 "my dog is cute" 转换为句子 "my dog is [Mask]"。
    + 10%的时候将其替换为其它任意单词，将单词 "hairy" 替换成另一个随机词，例如 "apple"。将句子 "my dog is cute" 转换为句子 "my dog is apple"。
    + 10%的时候会保留原始Token，例如保持句子为 "my dog is cute" 不变。。

+ Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。如果是的话输出’IsNext‘，否则输出’NotNext‘。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在图4中的[CLS]符号中。

 |输入 = [CLS] 我 喜欢 玩 [Mask] 联盟 [SEP] 我 最 擅长 的 [Mask] 是 亚索 [SEP]|
 |----|
 | 类别 = IsNext |

 | 输入 = [CLS] 我 喜欢 玩 [Mask] 联盟 [SEP] 今天 天气 很 [Mask] [SEP] |
 |----|
 | 类别 = NotNext |

## BERT的微调

在海量的语料上训练完BERT之后，便可以将其应用到NLP的各个任务中了。
微调(Fine-Tuning)的任务包括：基于句子对的分类任务，基于单个句子的分类任务，问答任务，命名实体识别等。

+ 基于句子对的分类任务：
	+ MNLI：给定一个前提 (Premise) ，根据这个前提去推断假设 (Hypothesis) 与前提的关系。该任务的关系分为三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。所以这个问题本质上是一个分类问题，我们需要做的是去发掘前提和假设这两个句子对之间的交互信息。
    + QQP：基于Quora，判断 Quora 上的两个问题句是否表示的是一样的意思。
    + QNLI：用于判断文本是否包含问题的答案，类似于我们做阅读理解定位问题所在的段落。
    + STS-B：预测两个句子的相似性，包括5个级别。
    + MRPC：也是判断两个句子是否是等价的。
    + RTE：类似于MNLI，但是只是对蕴含关系的二分类判断，而且数据集更小。
    + SWAG：从四个句子中选择为可能为前句下文的那个。
+ 基于单个句子的分类任务
	+ SST-2：电影评价的情感分析。
    + CoLA：句子语义判断，是否是可接受的（Acceptable）。
+ 问答任务
	+ SQuAD v1.1：给定一个句子（通常是一个问题）和一段描述文本，输出这个问题的答案，类似于做阅读理解的简答题。
+ 命名实体识别
	+ CoNLL-2003 NER：判断一个句子中的单词是不是Person，Organization，Location，Miscellaneous或者other（无命名实体）。
    
<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/46789704fc834558b340e0328253108e66cac3ab7b784e31b01f393652d9ed55" width = "700"/> <br />
</p><br><center>图3 BERT 用于不同的 NLP 任务</center></br>
