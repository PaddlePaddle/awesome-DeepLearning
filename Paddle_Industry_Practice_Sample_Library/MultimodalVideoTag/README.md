# 多模态视频分类

## 内容

* [项目说明](#项目说明)

* [安装说明](#安装说明)

* [数据准备](#数据准备)

* [模型训练](#模型训练)

* [模型评估](#模型评估)

* [模型推理](#模型推理)

* [模型优化](#模型优化)

* [模型部署](#模型部署)

* [参考论文](#参考论文)

  <a name="项目说明"></a>

## 1 项目说明

随着UGC视频的爆炸增长，短视频人均使用时长及头部短视频平台日均活跃用户均持续增长，内容消费的诉求越来越受到人们的重视。同时对视频内容的理解丰富度要求也越来越高，需要对视频所带文本、音频、图像多模态数据多角度理解，才能提炼出用户真实的兴趣点和高层次语义信息。目前存在以下挑战：

* 标签高语义
* 模态不对齐
* 多模态语义鸿沟

我们使用MutimodalVideoTag多模态视频分类模型，基于Paddle2.0版本进行开发。模型基于真实短视频业务数据，融合文本、视频图像、音频三种模态进行视频多模标签分类，相比纯视频图像特征，显著提升高层语义标签效果。其原理示意如 **图1 **所示。

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/a1a6b7ace28a4e999a2630ea0b776ef5123fec36ec7b414b8cff6fc1cc74bb8e' width='700'></center>
<center>图1 MutimodalVideoTag 多模态视频分类模型示意图</center>

**欢迎报名直播课加入交流群，如需更多技术交流与合作可点击[报名链接](https://paddleqiyeban.wjx.cn/vj/Qlb0uS3.aspx?udsid=531417)**

<a name="安装说明"></a>

## 2 安装说明

环境要求
* PaddlePaddle>=2.0.0
* Python >= 3.5

下载PaddleVideo源码，**下载一次**即可：

```python
!git clone https://github.com/PaddlePaddle/PaddleVideo.git
```

* 注：更多安装教程请参考[安装教程](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/install.md)

<a name="数据准备"></a>

## 3 数据准备

数据源来自UGC用户制作上传客户视频和视频标题，分别对视频三个模态的数据进行处理，对视频进行抽帧，获得图像序列；抽取视频的音频pcm文件；收集视频标题，简单进行文本长度截断，一般取50个字。

本项目提供已经抽取好图像、音频特征的特征文件，以及标题和标签信息，模型方面提供训练好checkpoint 文 件，可进行finetune、模型评估、预测。执行如下命令即可下载，**数据下载一次即可**。


```python
%cd /home/aistudio/PaddleVideo/applications/MultimodalVideoTag/
```


```python
!sh download.sh
```

下载的数据文件包括抽取好特征的文件夹`feature_files`，以及记录划分的txt 文件。其中`feature_files`包含了特征文件，格式为pkl，我们通过读取pkl文件查看存储结构。


```python
import pickle
import numpy as np
record = pickle.load(open('datasets/feature_files/74f021eb0d34ac98bc47837f9cbc5afa.pkl', 'rb'))
print(record.keys())
print('video feature: ', np.array(record['feature']['image_pkl']).shape)
print('audio feature: ', np.array(record['feature']['audio_pkl']).shape)
print('label: ', record['label'])
```

    dict_keys(['video', 'feature', 'label'])
    video feature:  (59, 2048)
    audio feature:  (60, 128)
    label:  [12]


记录划分的txt 文件格式如下：
```
文件名 \t 标题 \t 标签
18e9bf08a2fc7eaa4ee9215ab42ea827.mp4 叮叮来自肖宇梁肖宇梁rainco的特别起床铃声 拍人-帅哥,拍人-秀特效,明星周边-其他明星周边
```
我们可以通过`head`命令展示几条数据。

```python
!head -n 10 datasets/val.txt
```

    ffce63f737137cab7b50126c10f636e3.mp4	小俊00001我的世界如果一个女的很喜欢你但你拒绝了会怎么样	游戏-沙盒
    caf17db009b7dcc77b4716384efb320e.mp4	和平精英这游戏玩的太难了	游戏-射击
    e977bd4ddd022624b74e6a0d1f366aad.mp4	女神	拍人-美女
    a2c66030625494b7aa45648e4c445805.mp4	过年啦	美食-美食展示
    f7fe09e2734eb5ad84595a821d0db6b8.mp4	新神第五人格先知我活得像个魔术师	游戏-角色扮演
    d57b2a72267aee3d83df645663355a5b.mp4	你想去哪都可以有个条件就是你是我的	游戏-MOBA
    a569d6e829baa41129ea8c78baa92601.mp4	皇室战争里程碑的纪念莽近200	游戏-策略游戏
    533b586f5426425f979ccabc68cfda77.mp4	这都是谁呢	拍人-萌娃
    a96197ffe31a810012eb83881ab3856f.mp4	比起扎头发女生披肩散发更让男生心动	拍人-美女
    2cd7fe0e846ac246eaad54cfaaaf2715.mp4	和平精英和平精英搞笑视频太难了	游戏-射击

```python
# 标签文件
!head -n 10 datasets/class.txt
```

    拍人-萌娃
    拍人-秀特效
    拍人-美女
    拍人-视频直播
    拍人-帅哥
    拍人-秀身材
    拍人-秀恩爱
    拍人-秀拍摄技巧
    游戏-休闲益智类游戏
    游戏-游戏周边

<a name="模型训练"></a>

## 4 模型训练

模型训练整体流程如 **图2** 所示，

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/b637280a4ece4518be7ad6258a95339b398035cf9a30440aad1baf36e75486ad' width='700'></center>
 <center>图2 模型训练流程图</center>

包含以下几个步骤：
* 特征抽取：使用预训练的 ResNet 对图像抽取高层语义特征；使用预训练的VGGish网络抽取音频特征；文本方面使用[ERNIE 1.0](https://github.com/PaddlePaddle/ERNIE)抽取文本特征，无需预先抽取，支持视频分类模型finetune
* 序列学习：分别使用独立的LSTM 对图像特征和音频特征进行序列学习，文本方面预训练模型对字符序列进行建模，在ernie 后接入一个textcnn 网络做下游任务的迁移学习。
* 多模融合：文本具有显式的高层语义信息，将文本特征引入到LSTM pooling 过程指导图像和音频时序权重分配，进行交叉融合，最后将文本、音频、视频特征拼接。
* 预测结果：分类器选用sigmoid 多标签分类器，支持视频多标签输出。

**文本特征提取**

纯短文本分类，仅使用视频文本情况下(这里来自视频标题)，评估多种文本编码器的分类效果，基于预训练模型的Bert和ERNIE有较大优势。

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/8dea935c200444ada081e7c8bfe1060fed0ea2ba01a34c5aba7d3454b47dd050' width='700'></center>
<center>图3 文本特征提取示意图</center>

**图像特征提取**

对视频进行截帧，每帧图像特征取自训练好的图像实体分类特征层，即分类前的隐层向量2048维， 作为图像的高语义向量表示，图像实体分类在千万规模、1000+类别图片上使用RestNet 进行训练。

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/bb12f7cdae1445d89a0ba7b2e45754ea381fd13b8b7d41178cd7cb9e31965353' width='700'></center>
<center>图4 图像特征提取示意图</center>

**音频特征提取**

对视频抽取音频wav文件，整个音频文件以960ms分段，每个960ms作为一个分段抽取128纬特征，960ms内部以窗口大小25ms，步长大小10ms ，通过mel + log传统音频特征抽取，抽取96* 64的音频特征，把这个特征当做一幅长度96，宽度64的图输入到VGGish 深度CNN 网络提取128纬特征向量，该网络使用Audioset(大规模音频分类数据集) 进行预训练。

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/376f1678e07e4b7caf80b8d3b286b58b59272365202e4c96b80cf8ca864edb8c' width='700'></center>

<center>图5 音频特征提取示意图</center>

**多模态融合**

融合图像、音频、文本 三种模态特征，图像和音频为时间可对齐的序列特征，而和文本的序列非对齐， 结合UGC 视频主题杂乱的特点，将文本特征融入到音频、文本的时序attention中，增强文本和特定视频 帧、音频特征的匹配作用。也尝试基于门控的GMU方法，利用门控分别对每个模态控制，最后模态叠加。

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/72c9ef6e7e134758ad0c73629d407bd8c36fe220de564a209dfa4d5504031a73' width='700'></center>

<center>图6 多模态融合示意图</center>

模型训练过程有如下可调模式，可在根据数据集情况进行调整，在conf/conf.txt 文件中

* ernie_freeze: 用于控制文本提特征的ernie 网络是否进行finetune，因为ernie 复杂度远大于图像、视频序列学习网络，因此在某些数据集上不好训练。
* lstm_pool_mode: 用于控制lstm 序列池化的方式，默认是"text_guide"表示利用文本加强池化注意力权重，如果设置为空，则默认为自注意力的权重。

执行如下命令启动训练即可。


```python
%cd /home/aistudio/PaddleVideo/applications/MultimodalVideoTag/
!sh train.sh
```

<a name="模型评估"></a>

## 5 模型评估

模型对测试集进行评估，同时支持将checkpoint 模型转为inference 模型， 可用参数'save_only' 选项控制，设置即只用于做模型转换，得到inference 模型，输出结果第一行表示loss，第二行表示Hit@1 acc指标。


```python
!sh eval_and_save_model.sh
```

    W0209 18:50:57.891755  5341 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0209 18:50:57.896518  5341 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    load video 50
    33.17947191238403
    64.0



```python
!ls checkpoints_save_new/
```

    AttentionLstmErnie_epoch0_acc72.0  AttentionLstmErnie_epoch3_acc96.0
    AttentionLstmErnie_epoch1_acc84.0  AttentionLstmErnie_epoch4_acc100.0



```python
# 选择最优模型进行预测
!python scenario_lib/eval_and_save_model.py --model_name=AttentionLstmErnie \
--config=./conf/conf.txt \
--save_model_param_dir=checkpoints_save_new/AttentionLstmErnie_epoch4_acc100.0 \
--save_inference_model=inference_models_save
```

    W0210 15:15:17.022603  1194 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0210 15:15:17.027884  1194 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    load video 50
    4.129207887649536
    100.0

<a name="模型推理"></a>

## 6 模型推理

通过上一步得到的inference 模型进行预测，结果默认阈值为0.5，存储到json 文件中，在conf/conf.txt 文件 threshold 参数进行控制多标签输出的阈值。


```python
!sh inference.sh
```

查看推理结果，`"video_id"`表示测试的视频，`"labels"`分别表示预测类型和置信度。


```python
! head -n 10 output.json
```

    [
        {
            "video_id": "https://bj.bcebos.com/mct-imagedb/VideoClassify/short_video_mp4/yidian_2w_20200207/ffce63f737137cab7b50126c10f636e3.mp4",
            "labels": [
                [
                    "游戏-沙盒",
                    0.6376440525054932
                ]
            ]
        },

**<a name="模型优化"></a>**

## 7 模型优化


主要在文本分支进行了实验，首先加入文本对模型效果有明显提升，实验结果显示ERNIE 在多分支下不 微调，而是使用后置网络进行微调，训练速度快，且稳定，同时attention 方面使用文本信息增强图像、 音频的attention 学习能一定程度提升模型效果。
<center>

| 模型                                                         | Hit@1 | Hit@2 |
| ------------------------------------------------------------ | ----- | ----- |
| 图像+音频                    | 63 | 78 |
| 图像+音频+文本分支ERNIE 不finetune +self-attention                     | 71.07 | 83.72 |
| 图像+音频+文本分支ERNIE 不finetune +textcnn finetune + self-attention  | 72.66 | 85.01 |
| 图像+音频+文本分支ERNIE 不finetune +textcnn finetune + text-guide-attention | 73.29 | 85.59 |
|</center>|||

这里对多模融合方式进行实验，可以看到在同样没有拼接文本特征的情况下，使用文本进行指导video 和 audio 的 pooling 过程，仅仅只是贡献了LSTM pooling 的attention 权重，显著提升了模型效果+2.6%， 证明文本还是可以和图像，音频产生语义对齐的关系。下图为 attention 权重在时间上分布。
<center>

| 模型                                                         | Hit@1 | Hit@2 |
| ------------------------------------------------------------ | ----- | ----- |
| 图像+音频                    | 63 | 78 |
| 图像+音频+text-guide-attention                    | 66 | 80.5 |
|</center>|||

<center><img src='https://ai-studio-static-online.cdn.bcebos.com/075ad24a578c42f0bbcaad705c88547489fcbaade0a9408788e95c0277227555' width='700'></center>

模型鲁棒性，标题信息是不可控的，为了应对标题缺失的情况，这里的目标是不带标题信息，起 码应该达到video + audio 的效果，实验发现以整体概率置空效果较好，实验阈值整体概率值为 0.4时达到最佳。
<center>

|  模型 |  预测不带标题Hit@1 |  带标题Hit@1 |  带标题Hit@2 |
| -------- | -------- | -------- |-------- |
|   图像+音频   |   63   |    63  |    78  |
|   图像+音频+文本   |   55   |    73.3  |    85.6  |
|   图像+音频+随机0.4 drop   |   62   |    72.4  |    85.1  |
</center>

<a name="模型部署"></a>

## 8 模型部署

在项目中为用户提供了基于paddle inference C++ 接口部署的方案，效果如下所示。用户可根据实际情况自行参考。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c9b31158d77047eeb32c9ab979ead46d294b49d2522b47bc8e404a103503fe61" width="700"/>
</div>

**欢迎报名直播课加入交流群，如需更多技术交流与合作可点击[报名链接](https://paddleqiyeban.wjx.cn/vj/Qlb0uS3.aspx?udsid=531417)**

<a name="参考论文"></a>

## 参考论文

* [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
* [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan
* [Ernie: Enhanced representation through knowledge integration](https://arxiv.org/abs/1904.09223), Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Chen, Xuyi and Zhang, Han and Tian, Xin and Zhu, Danxiang and Tian, Hao and Wu, Hua

# 资源
更多资源请参考：

* 更多深度学习知识、产业案例，请参考：[awesome-DeepLearning](https://github.com/paddlepaddle/awesome-DeepLearning)

* 更多动作识别、动作检测、多模态、视频目标分割、单目深度估计模型，请参考：[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)

* 更多学习资料请参阅：[飞桨深度学习平台](https://www.paddlepaddle.org.cn/?fr=paddleEdu_aistudio)
