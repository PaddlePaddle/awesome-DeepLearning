# 基于百度自研模型ERNIE进行事件抽取任务


信息抽取旨在从非结构化自然语言文本中提取结构化知识，如实体、关系、事件等。事件抽取是信息抽取的一种，其目标是对于给定的自然语言句子，根据预先指定的事件类型和论元角色，识别句子中所有目标事件类型的事件，并根据相应的论元角色集合抽取事件所对应的论元。其中目标事件类型 (event_type) 和论元角色 (role) 限定了抽取的范围。

**图1**展示了一个关于事件抽取的样例，可以看到原句子描述中共计包含了2个事件类型event_type：胜负和夺冠，其中对于胜负事件类型，论元角色role包含时间，胜者，败者，赛事名称；对于夺冠事件类型，论元角色role包含夺冠事件，夺冠赛事，冠军。总而言之，事件抽取期望从这样非结构化的文本描述中，提取出事件类型和元素角色的结构化信息。

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/8df72cd00e684ee2b274696b20c64111a98e93d1dbe74ee8875e3c39cc8f4978" alt="事件抽取" align=center />
</div>
<center>图1 事件抽取样例</center>

本案例将基于ERNIE模型，在[DuEE 1.0](https://aistudio.baidu.com/aistudio/competition/detail/65) 数据集上进行事件抽取任务。

## 1. 方案设计
本实践设计方案如图2所示，本案例将采用分阶段地方式，分别训练触发词识别和事件元素识别两个模型去抽取对应的触发词和事件元素。模型的输入是一串描述事件的文本，模型的输出是从事件描述中提取的事件类型，事件元素等信息。

具体而言，在建模过程中，对于输入的待分析事件描述文本，首先需要进行数据处理生成规整的文本序列数据，包括语句分词、将词转换为id，过长文本截断、过短文本填充等等操作；然后，将规整的数据传到触发词识别模型中，识别出事件描述中的触发词，并且根据触发词判断该事件的类型；接下来，将规整的数据继续传入事件元素识别模型中，并确定这些事件元素的角色；最后将两个模型的输出内容进行汇总，获得最终的提取事件结果，其将主要包括事件类型，事件元素和事件角色。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/96d1d2a6c6a54d51a9f22b0c6d9680c92f33779f2c384e55a84aff1103ea88b6" /></center>
<center>图2 事件提取设计方案</center>
<br/>

其中本案例中我们将触发词识别模型和事件元素模型定义为序列标注任务，两者均将采用ERNIE模型完成数据标注任务，从而分别抽取出事件类型和事件元素，后续会将两者的结果进行汇总，得到最终的事件提取结果。

对于触发词抽取模型，该部分主要是给定事件类型，识别句子中出现的事件触发词对应的位置以及对应的事件类别，模型原理图如下：

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/435eb3cde281427eaefedf942dbdd425e8de5e2790884f5ebc16749fbda7b609" width="500" height="400" alt="基于序列标注的触发词抽取模型" align=center />
</div>
<center>图3 触发词抽取模型图</center>
<br/>

可以看到上述样例中通过模型识别出：1）触发词"收购"，并分配标签"B-收购"、"I-收购"。同样地，对于论元抽取模型，该部分主要是识别出事件中的论元以及对应论元角色，模型原理图如下：

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/6c47ba6465784fd0a715e86c2916b943fb48e709b4104d69ab9c39cb000929a7" width="500" height="400" alt="基于序列标注的论元抽取模型" align=center />
</div>
<center>图4 论元抽取模型</center>
<br/>

可以看到上述样例中通过模型识别出：1）触发词"新东方"，并分配标签"B-收购方"、"I-收购方"、"I-收购方"；2）论元"东方优播", 并分配标签"B-被收购方"、"I-被收购方"、"I-被收购方"、"I-被收购方"。


## 2. 数据说明
[DuEE 1.0](https://aistudio.baidu.com/aistudio/competition/detail/65) 是百度发布的中文事件抽取数据集，包含65个事件类型的1.7万个具有事件信息的句子（2万个事件）。事件类型根据百度风云榜的热点榜单选取确定，具有较强的代表性。65个事件类型中不仅包含「结婚」、「辞职」、「地震」等传统事件抽取评测中常见的事件类型，还包含了「点赞」等极具时代特征的事件类型。具体的事件类型及对应角色见表3。数据集中的句子来自百度信息流资讯文本，相比传统的新闻资讯，文本表达自由度更高，事件抽取的难度也更大。

在实验之前，请确保下载DuEE1.0数据，并将其解压后的如下四个数据文件放在`./dataset`目录下：
- duee_train.json: 原训练集数据文件
- duee_dev.json: 原开发集数据文件
- duee_test.json: 原测试集数据文件
- duee_event_schema.json: DuEE1.0事件抽取模式文件，其定义了事件类型和事件元素角色等内容

其中单条样本的格式如下所示：
```
{
    "text":"华为手机已经降价，3200万像素只需千元，性价比小米无法比。",
    "id":"2d41b63e42127b9e8e0416484e9ebd05",
    "event_list":[
        {
            "event_type":"财经/交易-降价",
            "trigger":"降价",
            "trigger_start_index":6,
            "arguments":[
                {
                    "argument_start_index":0,
                    "role":"降价方",
                    "argument":"华为",
                    "alias":[

                    ]
                },
                {
                    "argument_start_index":2,
                    "role":"降价物",
                    "argument":"手机",
                    "alias":[

                    ]
                }
            ],
            "class":"财经/交易"
        }
    ]
}
```
**备注**：可点击 [DuEE 1.0](https://aistudio.baidu.com/aistudio/competition/detail/65) 进行数据下载

## 3. 使用说明
### 3.1 模型训练
可按如下方式，使用训练集对触发词和元素识别模型进行训练。

训练触发词识别模型
```shell
sh run_train.sh trigger
```
训练元素识别模型
```shell
sh run_train.sh role
```

### 3.2 模型评估
可按如下方式，使用测试集对触发词和元素识别模型进行评估。

测试触发词识别模型
```shell
sh run_evaluate.sh trigger
```
测试元素识别模型
```shell
sh run_evaluate.sh role
```
### 3.3 模型推理
使用如下命令进行模型测试。
```shell
sh run_predict.sh
```