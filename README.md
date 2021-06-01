# 一、简介

本项目是深度学习赋能材料获取一站式平台，内容涵盖深度学习教程、理论知识点解读、产业实践案例、常用Tricks和前沿论文复现等。从理论到实践，从科研到产业应用，各类学习材料一应俱全，旨在帮助开发者高效学习和掌握深度学习知识，快速成为AI跨界人才。

* **内容全面**：无论您是深度学习初学者，还是资深用户，都可以在本项目中快速获取到需要的学习材料。

* **形式丰富** ：赋能材料形式多样，包括可在线运行的notebook、视频、书籍、B站直播等，满足您随时随地学习的需求。

* **实践代码实时更新**：本项目中涉及到的代码均匹配Paddle最新发布版本，开发者可以实时学习最新的深度学习任务实现方案。

* **前沿知识分享** ：定期分享顶会最新论文解读和代码复现，开发者可以实时掌握最新的深度学习算法。



## 最新动态

2021年5月14日-5月20日，B站《零基础实践深度学习》7日打卡课，扫描下方二维码快速入群，了解最新的课程信息。

<center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/readme/qr_code.png?raw=true"/></center>
<br></br>



# 二、内容概览

## 1. 零基础实践深度学习

  - **[AI Studio在线课程：《零基础实践深度学习》](https://aistudio.baidu.com/aistudio/course/introduce/1297
    )**：理论和代码结合、实践与平台结合，包含20小时视频课程，由百度杰出架构师、飞桨产品负责人和资深研发人员共同打造。

    <center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/readme/aistudio.png?raw=true"/></center><br></br>


  - **《零基础实践深度学习》书籍**：由清华出版社2020年底发行，京东/当当等电商均有销售。

    <center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/readme/book.png?raw=true"/></center><br></br>
    
    

## 2. [深度学习百科及面试资源](https://paddlepedia.readthedocs.io/en/latest/index.html)

* [深度学习](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/index.html#)
  * [基础知识](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/basic_concepts/index.html)（包括神经元，单层感知机，多层感知机等5个知识点）
  * [优化策略](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/optimizers/index.html)（包括什么是优化器,GD，SGD，BGD,鞍点,Momentum,NAG,Adagrad,AdaDelta,RMSProp,Adam,AdaMax,Nadam,AMSGrad,AdaBound,AdamW,RAdam,Lookahead等18个知识点）
  * [激活函数](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/index.html)(包括什么是激活函数、激活函数的作用、identity、step、sigmoid、tanh、relu、lrelu、prelu、rrelu、elu、selu、softsign、softplus、softmax、swish、hswish、激活函数的选择等21个知识点)
  * [常用损失函数](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/index.html)（包括交叉熵损失、MSE损失以及CTC损失等3个知识点）
  * [模型调优](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/index.html#)
     * [学习率](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/learning_rate.html)（包括什么是学习率、学习率对网络的影响以及不同的学习率率衰减方法，如：分段常数衰减等12个学习率衰减方法）
     * [归一化](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/index.html)（包括什么是归一化、为什么要归一化、为什么归一化能提高求解最优解速度、归一化有哪些类型、不同归一化的使用条件、归一化和标准化的联系与区别等6个知识点）
     * [正则化](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/regularization/index.html)(包括什么是正则化？正则化如何帮助减少过度拟合？数据增强，L1 L2正则化介绍，L1和L2的贝叶斯推断分析法，Dropout，DropConnect,早停法等8个知识点) 	
     * [注意力机制]() (包括自注意力，多头注意力，经典注意力计算方式等10个知识点)
     * [Batch size](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/batch_size.html)（包括什么是batch size、batch size对网络的影响、batch size的选择3个知识点）
     * [参数初始化](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/weight_initializer.html)（包括为什么不能全零初始化、常见的初始化方法等5个知识点）
* [计算机视觉](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/index.html)
  * [卷积算子](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/convolution_operator/index.html)（包括标准卷积、1*1卷积、3D卷积、转置卷积、空洞卷积、分组卷积、可分离卷积等7个知识点）
  * [图像增广](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/image_augmentation/index.html)（包括什么是数据增广、常用数据增广方法、图像变换类增广方法、图像裁剪类增广方法、图像混叠类增广方法、不同方法对比实验等11个知识点）
  * [卷积神经网络](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/convolution_network/index.html)（包括CNN综述、池化等2个知识点）
  * [图像分类](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/classification/index.html)（包括LeNet、AlexNet、VGG、GoogleNet、DarkNet、ResNet等6个知识点）
  * [目标检测](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/index.html)（包括目标检测综述、边界框、锚框、交并比、NMS、IOU等6个知识点）
* [自然语言处理](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/index.html)
  * [词表示](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/word_representation/index.html) (包括one-hot编码，word-embedding,word2vec等9个知识点)
  * [序列模型](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/sequence_model/index.html)（包括GRU，LSTM，RNN等3个知识点）
  * [命名实体识别](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/index.html) (包括bilstm+crf架构剖析，crf原理等8个知识点)
* [预训练模型](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/index.html) 
  * [预训练模型是什么](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/pretrain_model_description.html) (包括预训练，微调等2个知识点)
  * [预训练分词Subword](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/subword.html)(包括BPE，WordPiece，ULM等3个知识点)
  * [Transformer](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/transformer.html)（包括self-attention,multi-head Attention,Position Encoding, Transformer Encoder, Transformer Decoder等5个知识点）
  * [BERT](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html)（包括BERT预训练任务，BERT微调等2个知识点）
  * [ERNIE](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/erine.html)(包括ERNIE介绍，Knowledge Masking等2个知识点)
* [推荐系统](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/index.html)
  * [推荐系统基础](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/recommender_system.html)(包括协同过滤推荐，内容过滤推荐，组合推荐，用户画像，召回，排序等6个知识点)
  * [DSSM模型](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/dssm.html)（包括DSSM模型等1个知识点）
* [对抗神经网络](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/index.html)
  * [encoder-decoder](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/encoder_decoder/index.html)
  * [GAN基本概念](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/index.html)
  * [GAN评价指标](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/gan_metric/index.html)
  * [GAN应用](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/gan_applications/index.html)
* [强化学习](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/index.html)
  * [强化学习基础知识点](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/basic_information.html)（包括智能体、环境、状态、动作、策略和奖励的定义）
  * [马尔可夫决策过程](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/markov_decision_process.html)
  * [策略梯度定理](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/policy_gradient.html)
  * [蒙特卡洛策略梯度定理](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/policy_gradient.html)
  * [REINFORCE算法](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/policy_gradient.html#reinforce)
  * [SARSA](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/Sarsa.html)（包括SARSA的公式，优缺点等2个知识点）
  * [Q-Learning](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/Q-learning.html)（包括Q-Learning的公式，优缺点等2个知识点）
  * [DQN](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/DQN.html#)（包括DQN网络概述及其创新点和算法流程2个知识点）

详细信息请参阅[Paddle知识点文档平台](https://paddlepedia.readthedocs.io/en/latest/index.html)


## 3. 产业实践深度学习（开发中）
## 4. Transformer系列特色课（开发中）
  

# 三、技术交流

非常感谢您使用本项目。您在使用过程中有任何建议或意见，可以在 **[Issue](https://github.com/PaddlePaddle/tutorials/issues)** 上反馈给我们，也可以通过扫描下方的二维码联系我们，飞桨的开发人员非常高兴能够帮助到您，并与您进行更深入的交流和技术探讨。

<center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/readme/qr_code.png?raw=true"/></center><br></br>



# 四、许可证书

本项目的发布受[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.txt)许可认证。



# 五、贡献内容

本项目的不断成熟离不开各位开发者的贡献，如果您对深度学习知识分享感兴趣，非常欢迎您能贡献给我们，让更多的开发者受益。

