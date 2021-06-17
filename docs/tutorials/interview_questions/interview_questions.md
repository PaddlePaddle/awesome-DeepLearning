# 深度学习基础

* [为什么归一化能够提高求解最优解的速度？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/basic_normalization.html#id4)
* [为什么要归一化？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/basic_normalization.html)
* [归一化与标准化有什么联系和区别？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/basic_normalization.html#id7)
* [归一化有哪些类型？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/basic_normalization.html#id5)
* [Min-max归一化一般在什么情况下使用？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/basic_normalization.html#id6)
* [Z-score归一化在什么情况下使用？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/normalization/basic_normalization.html#id6)
* [学习率过大或过小对网络会有什么影响？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/learning_rate.html)
* [batch size的大小对网络有什么影响？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/batch_size.html)
* [在参数初始化时，为什么不能全零初始化？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/weight_initializer.html)
* [激活函数的作用？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/activation_functions/Activation_Function.html#id3)
* [sigmoid函数有什么优缺点？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/activation_functions/Activation_Function.html#sigmoid)
* [RELU函数有什么优缺点？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/activation_functions/Activation_Function.html#relu)
* [如何选择合适的激活函数？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/activation_functions/Activation_Function.html#id5)
* [为什么 relu 不是全程可微/可导也能用于基于梯度的学习？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/activation_functions/Activation_Function.html#id6)
* [怎么计算mAP？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/metrics/mAP.html)
* [交叉熵为什么可以作为分类任务的损失函数？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/CE_Loss.html)
* [CTC方法主要使用了什么方式来解决了什么问题？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/CTC.html#)
* [机器学习指标精确率，召回率，f1指标是怎样计算的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/metrics/evaluation_metric.html)

# 卷积模型

- [相较于全连接网络，卷积在图像处理方面有什么样的优势？](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Convolution.html#id1)
- [卷积中感受野的计算方式？](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Convolution.html#receptive-field)
- [1*1卷积的作用是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/1%2A1_Convolution.html)
- [深度可分离卷积的计算方式以及意义是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Separable_Convolution.html#id4)

# 预训练模型
* [BPE生成词汇表的算法步骤是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/subword.html#byte-pair-encoding-bpe)
* [Multi-Head Attention的时间复杂度是多少？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/transformer.html#multi-head-attention)
* [Transformer的权重共享在哪个地方？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/transformer.html#id6)
* [Transformer的self-attention的计算过程是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/transformer.html#self-attention)
* [讲一下BERT的基本原理](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#id1)
* [讲一下BERT的三个Embedding是做什么的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#embedding)
* [BERT的预训练做了些什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#id11)
* [BERT,GPT,ELMO的区别](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#id11)
* [请列举一下BERT的优缺点](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#id13)
* [ALBERT相对于BERT做了哪些改进？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/ALBERT.html#id2)
* [NSP和SOP的区别是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/ALBERT.html#sentence-order-prediction)

# 对抗神经网络

* [GAN是怎么训练的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/GAN%20train.html)

* [GAN生成器输入为什么是随机噪声](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/Input%20noise.html#gan)

* [GAN生成器最后一层激活函数为什么通常使用tanh()？](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/Generator.html#generator)

* [GAN使用的损失函数是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/GAN%20loss.html)

* [GAN中模式坍塌(model callapse指什么？)](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/Collapse.html)

* [GAN模式坍塌解决办法](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/Collapse.html)

* [GAN模型训练不稳定的原因](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/Unstable%20training.html#)

* [GAN模式训练不稳定解决办法 or 训练GAN的经验/技巧](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/basic_concept/Unstable%20training.html#)

# 计算机视觉

* [ResNet中Residual block解决了什么问题？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/classification/ResNet.html)
* [使用Cutout进行数据增广有什么样的优势？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/image_augmentation/ImageAugment.html#cutout)
* [GoogLeNet使用了怎样的方式进行了网络创新？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/classification/GoogLeNet.html)
* [ViT算法中是如何将Transformer结构应用到图像分类领域的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/classification/ViT.html)
* [NMS的原理以及具体实现？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/NMS.html)
* [OCR常用检测方法有哪几种、各有什么优缺点](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR.html#id2)

* [介绍一下DBNet算法原理](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR_Detection/DBNet.html#id3)
* [DBNet 输出是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR_Detection/DBNet.html#id2)
* [DBNet loss](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR_Detection/DBNet.html#loss)
* [介绍以下CRNN算法原理](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR_Recognition/CRNN.html#crnn)
* [介绍一下CTC原理](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/CTC.html)
* [OCR常用的评估指标](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR.html#id7)

* [OCR目前还存在哪些挑战/难点？](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/OCR/OCR.html#id9)

# 自然语言处理
* [RNN一般有哪几种常用建模方式?](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/rnn.html#span-id-4-rnn-span)
* [LSTM是如何改进RNN，保持长期依赖的?](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/lstm.html#span-id-1-lstm-span)
* [LSTM在每个时刻是如何融合之前信息和当前信息的?](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/lstm.html#span-id-3-lstm-span)
* [使用LSTM如何简单构造一个情感分析任务?](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/lstm.html#span-id-4-lstm-span)
* [介绍一下GRU的原理](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/gru.html)
* [word2vec提出了哪两种词向量训练方式](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/word_representation/word2vec.html#id1)
* [word2vec提出了负采样的策略，它的原理是什么，解决了什么样的问题？](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/word_representation/word2vec.html#skip-gram)
* [word2vec通过什么样任务来训练词向量的?](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/word_representation/word2vec.html#)
* [如果让你实现一个命名实体识别任务，你会怎么设计?](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html#1)
* [在命名实体识别中，一般在编码网络的后边添加CRF层有什么意义](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html#1)
* [介绍一下CRF的原理](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html#2.1)
* [CRF是如何计算一条路径分数的?](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html#2.4)
* [CRF是如何解码序列的?](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html#2.6)
* [使用bilstm+CRF做命名实体识别时，任务的损失函数是怎么设计的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/ner/bilstm_crf.html#2.3)
* [BERT的结构和原理是什么?](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#id1)
* [BERT使用了什么预训练任务?](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html#id11)
* [说一下self-attention的原理?](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/transformer.html#self-attention)

# 推荐系统

* [DSSM模型的原理是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/dssm.html)
* [DSSM怎样解决OOV问题的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/dssm.html#id2)
* [推荐系统的PV和UV代表什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/evaluation_metric.html#id2)
* [协同过滤推荐和基于内容的推荐的区别是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/evaluation_metric.html#id2)
* [说一说推荐系统的交叉验证的方法？](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/evaluation_metric.html#id2)

# 模型压缩

* [为什么需要进行模型压缩？](https://paddlepedia.readthedocs.io/en/latest/tutorials/model_compress/model_compress.html)

* [模型压缩的基本方法有哪些？](https://paddlepedia.readthedocs.io/en/latest/tutorials/model_compress/model_compress.html#id3)

* [DynaBERT模型的创新点是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/model_compress/model_distill/DynaBERT.html)

* [TinyBERT是如何对BERT进行蒸馏的？](https://paddlepedia.readthedocs.io/en/latest/tutorials/model_compress/model_distill/TinyBERT.html)

  

# 强化学习

* [DQN网络的创新点是什么？](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/DQN.html#id1)

* [什么是马尔可夫决策过程？](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/basic_information.html)
* [什么是SARSA？](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/Sarsa.html#id1)
* [什么是Q-Learning？](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/Q-learning.html#id1)

  

