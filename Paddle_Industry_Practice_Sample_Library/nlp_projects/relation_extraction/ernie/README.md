# 基于预训练模型完成实体关系抽取

信息抽取旨在从非结构化自然语言文本中提取结构化知识，如实体、关系、事件等。实体关系抽取是信息抽取的一种，其目标是对于给定的自然语言句子，根据预先定义的schema集合抽取出所有满足schema约束的SPO三元组，即<subject, predicate, object>。其中schema定义了关系P以及其对应的主体S和客体O的类别。因此，实体关系抽取的输入和输出分别是：
- 输入：一个或多个连续完整句子。
- 输出：句子中包含的所有符合给定schema约束的SPO三元组。

这里需要注意的是，DuIE 2.0数据集中存在两种object类型:简单O值和复杂O值， 其中，简单O值特指O是一个单一的文本片段，该类型是最常见的关系类型；复杂O值特指O是一个结构体，由多个语义明确的文本片段共同组成，多个文本片段对应了结构体中的多个槽位 (slot)。下面分别给出了简单O值和复杂O值的两个类型描述。
1.简单O值：「妻子」关系的schema定义为：

```
{
    S_TYPE: 人物,
    P: 妻子,
    O_TYPE: {
        @value: 人物
    }
}
```

2.复杂O值：「饰演」关系中O值有两个槽位@value和inWork，分别表示「饰演的角色是什么」以及「在哪部影视作品中发生的饰演关系」，其schema定义为：
```
{
    S_TYPE: 娱乐人物,
    P: 饰演,
    O_TYPE: {
        @value: 角色,
        inWork: 影视作品
    }
}
```

本实践将基于ERNIE预训练模型，在百度数据集[DuIE 2.0](https://aistudio.baidu.com/aistudio/competition/detail/46) 上进行实体关系抽取任务。

## 1. 方案设计

本实践将基于ERNIE预训练模型进行实体关系抽取任务，如图2所示。在实体关系抽取任务建模的过程中，比较核心的一点是如何将以上提到的SPO模型映射为适合模型输入的形式。在实体关系抽取任务中，存在头实体subject，尾实体object以及预测关系predicate，这三项内容组成了完整的实体预测关系<subject, predicate, object>，即SPO。基于此我们设计了这样的一套标签体系：
- 使用S和O分别表示头实体subject和尾实体object，后续拼接关系predicate，最后拼接subject或者object类型，如S-配音-娱乐人物；
- 标注每个实体的起始标签：对于简单O值情况，一个SPO三元组可标注两种标签；对于复杂O值情况，由于存在多个O值，可标注一个头实体标签和多个尾实体标签；
- 对于每个实体的中间token，统一标注为I；
- 对于非实体的token，统一标注为O；

本实践中，首先将文本传入预训练模型ERNIE中进行编码，然后对每个token进行多标签分类任务，从而得到该token应该被赋予的标签分布，这里需要注意的一点是每个token均有可能被标注为多个标签。在获得每个token的标签分布之后，我们将根据标签分布进行解码，从而得到原始文本的SPO列表。

## 2. 数据说明
[DuIE 2.0](https://aistudio.baidu.com/aistudio/competition/detail/46) 是业界规模最大的基于schema的中文关系抽取数据集，包含超过43万三元组数据、21万中文句子及48个预定义的关系类型。数据集中的句子来自百度百科、百度贴吧和百度信息流文本。下面给出了一个DuIE 2.0的样例，可以看到，给定文本："王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人"，实体关系抽取期望能够抽取出文本中各实体之间的关系，如正大综艺主持人是王雪纯等。

```
{
    "text":"王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人",
    "spo_list":[
        {
            "predicate":"配音",
            "subject":"王雪纯",
            "subject_type":"娱乐人物",
            "object":{
                "@value":"晴雯",
                "inWork":"红楼梦"
            },
            "object_type":{
                "@value":"人物",
                "inWork":"影视作品"
            }
        },
        {
            "predicate":"主持人",
            "subject":"正大综艺",
            "subject_type":"电视综艺",
            "object":{
                "@value":"王雪纯"
            },
            "object_type":{
                "@value":"人物"
            }
        }
    ]
}
```

**备注**：可点击[DuIE 2.0](https://aistudio.baidu.com/aistudio/competition/detail/46) 进行数据下载。

## 3. 使用说明
### 3.1 模型训练
使用如下命令，进行模型训练。

```shell
sh run_train.sh
```

### 3.2 模型测试
使用如下命令，进行模型测试。

```shell
sh run_evaluate.sh 
```

### 3.3 模型推理
使用如下命令，进行模型推理。

```shell
sh run_predict.sh
```
