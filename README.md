

# 一、项目简介

本项目是[飞桨官方](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github)出品的一站式深度学习在线百科，飞桨致力于让深度学习技术的创新与应用更简单，更多飞桨内容欢迎访问[飞桨官网](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github)。本项目内容涵盖：

📒课程类：[**零基础实践深度学习**](https://aistudio.baidu.com/aistudio/course/introduce/1297)、**产业实践深度学习**、**[特色课程](https://aistudio.baidu.com/aistudio/education/group/info/24322)、飞桨套件课程汇总资料**

📒书籍类：**《动手学深度学习》paddle版**

📒宝典类：[**深度学习百问**](https://paddlepedia.readthedocs.io/en/latest/index.html)、**面试宝典**

📒案例类：**产业实践案例**

从理论到实践，从科研到产业应用，各类学习材料一应俱全，旨在帮助开发者高效地学习和掌握深度学习知识，快速成为AI跨界人才。

<center><img src="./docs/images/cover/repo_cover1.png" width=60%></center>

* **内容全面**：无论您是深度学习初学者，还是资深用户，都可以在本项目中快速获取到需要的学习材料。
* **形式丰富**：材料形式多样，包括可在线运行的notebook、视频、书籍、B站直播等，满足您随时随地学习的需求。
* **实时更新**：本项目中涉及到的代码均匹配Paddle最新发布版本，开发者可以实时学习最新的深度学习任务实现方案。
* **前沿分享**：定期分享顶会最新论文解读和代码复现，开发者可以实时掌握最新的深度学习算法。

#### <span id = '0'>如果本项目对您有帮助，欢迎点击网页右上方进行star❤️</span>

---

## 👨‍🏫我是高校用户

| 我希望：     | 我可以学习：                                                 |
| ------------ | ------------------------------------------------------------ |
| 入门深度学习 | 零基础实践深度学习[:arrow_heading_down:](#1)、深度学习百问[:arrow_heading_down:](#2)、动手学深度学习paddle版[:arrow_heading_down:](#dive) |
| 进阶深度学习 | 产业实践深度学习、深度学习百问[:arrow_heading_down:](#2)、面试宝典[:arrow_heading_down:](#6) |
| 趣味深度学习 | 特色课程[:arrow_heading_down:](#3)、产业实践案例[:arrow_heading_down:](#5) |

## 👷‍♂️我是企业用户

| 我希望：     | 我可以学习：                                                 |
| ------------ | ------------------------------------------------------------ |
| 入门深度学习 | 零基础实践深度学习[:arrow_heading_down:](#1)、深度学习百问[:arrow_heading_down:](#2)、动手学深度学习paddle版[:arrow_heading_down:](#dive) |
| 进阶深度学习 | 产业实践深度学习、特色课程[:arrow_heading_down:](#3)、面试宝典[:arrow_heading_down:](#6) |
| 实践深度学习 | 产业实践案例[:arrow_heading_down:](#5)、飞桨各产品课程[:arrow_heading_down:](#fj) |

---

# 二、项目内容

# 👉课程类

## <span id =1> 零基础实践深度学习</span>

  - **AI Studio在线课程：[《零基础实践深度学习》](https://aistudio.baidu.com/aistudio/course/introduce/1297
    )**：理论和代码结合、实践与平台结合，包含20小时视频课程，由百度杰出架构师、飞桨产品负责人和资深研发人员共同打造。

    <center><img src="./docs/images/cover/0_cover.png"/></center><br></br>


  - **《零基础实践深度学习》书籍**：本课程配套书籍，由清华出版社2020年底发行，京东/当当等电商均有销售。

    <center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/readme/book.png?raw=true"/></center><br></br>

## <span id ='3'>特色课 - Transformer系列</span>

| 章节名称                            | notebook链接                                                 | Python实现                                                   | 课程简介                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 经典的预训练语言模型(上)                | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2110336) | [Python实现](./transformer_courses/Transformer_Machine_Translation) | 本章节将为大家详细介绍NLP领域经典的模型word2vec，ELMo，Transformer |
| 经典的预训练语言模型（下）                | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2110336) | [Python实现](./transformer_courses/Transformer_Machine_Translation) | 本章节将为大家详细介绍NLP领域2个基于Transformer的预训练语言模型GPT，BERT，还会介绍Transformer在机器翻译里面的应用。 |
| 预训练模型在自然语言理解方面的改进  | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2166195) | [Python实现](./transformer_courses/reading_comprehension_based_on_ernie) | ERNIE， RoBERTa， KBERT，清华ERNIE等，在广度上去分析经典预训练模型的一些改进。 |
| 预训练模型在长序列建模方面的改进    | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2166197) | [Python实现](./transformer_courses/sentiment_analysis_based_on_xlnet) | Transformer-xl， xlnet， longformer等，分析BERT和transformer的长度局限，并讨论这些方法的改进点。 |
| BERT蒸馏                            | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2177549) | [Python实现](./transformer_courses/BERT_distillation)        | 本章节为大家详细介绍了针对BERT模型的蒸馏算法，包括：Patient-KD、DistilBERT、TinyBERT、DynaBERT等模型，同时以代码的形式为大家展现了如何使用DynaBERT的训练策略对TinyBERT进行蒸馏。 |
| 预训练模型的瘦身策略 – – 高效结构   | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2138857) | [Python实现](./transformer_courses/Transformer_Punctuation_Restoration) | 本章节将为大家详细介绍NLP领域，基于Transformer模型的瘦身技巧。包括 Electra，AlBERT 以及 performer。还会介绍代码实现案例：基于Electra的语音识别后处理中文标点符号预测 |
| transformer在图像分类中的应用(上)   | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2154618) | [Python实现](https://github.com/tngt/awesome-DeepLearning/blob/master/transformer_courses/Application_of_transformer_in_image_classification) | 本章节将为大家详细介绍 Transformer 在 CV 领域中的两个经典算法：ViT 以及 DeiT。带领大家一起学习Transformer 结构在图像分类领域的具体应用。 |
| transformer在图像分类中的应用（下） | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2280436) | [Python实现](https://github.com/tngt/awesome-DeepLearning/blob/master/transformer_courses/Application_of_transformer_in_image_classification_Swin) | 本章节将为大家详细介绍 Transformer 在 CV 领域中的经典算法：Swin Transformer。带领大家一起学习Transformer 结构在图像分类领域的具体应用。 |
| CV领域的Transformer模型DETR在目标检测任务中的应用 | [notebook链接](https://aistudio.baidu.com/aistudio/projectdetail/2290729) | [Python实现](./transformer_courses/object_detection_DETR) | 本章节将为大家详细介绍 Transformer 在目标检测领域中的经典算法：DETR。带领大家使用飞桨2.1版本在COCO数据集上实现基于DETR模型的目标检测，同时对训练好的模型进行评估和预测。 |

返回[:arrow_heading_up:](#0)

-----

# 👉书籍类

## <span id ='dive'>《动手学深度学习》paddle版</span>

[本项目](https://github.com/tngt/awesome-DeepLearning/tree/develop/Dive-into-DL-paddlepaddle)将《[动手学深度学习](http://zh.d2l.ai/)》原书中MXNet代码实现改为PaddlePaddle实现。原书作者：阿斯顿·张、李沐、扎卡里 C. 立顿、亚历山大 J. 斯莫拉以及其他社区贡献者，GitHub地址：https://github.com/d2l-ai/d2l-zh。

本项目面向对深度学习感兴趣，尤其是想使用PaddlePaddle进行深度学习的童鞋。本项目并不要求你有任何深度学习或者机器学习的背景知识，你只需了解基础的数学和编程，如基础的线性代数、微分和概率，以及基础的Python编程。

<div align=center>
<img width="500" src="./Dive-into-DL-paddlepaddle/docs/img/cover.jpg">
</div>
返回[:arrow_heading_up:](#0)


----

# 👉宝典类

## <span id ='2'>深度学习百问</span>

深度学习百问内容包含深度学习基础篇、深度学习进阶篇、深度学习应用篇、强化学习篇以及面试宝典，详细信息请参阅[Paddle知识点文档平台](https://paddlepedia.readthedocs.io/en/latest/index.html)。

* **深度学习基础篇**  

  1. [深度学习](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/index.html#)  

  2. [卷积神经网络](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/index.html)  

  3. [序列模型](https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/index.html)  

* **深度学习进阶篇**  

  1. [预训练模型](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/index.html)
  2. [对抗神经网络](https://paddlepedia.readthedocs.io/en/latest/tutorials/generative_adversarial_network/index.html)  

* **深度学习应用篇**  

  1. [计算机视觉](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/index.html)  
  2. [自然语言处理](https://paddlepedia.readthedocs.io/en/latest/tutorials/natural_language_processing/index.html)  
  3. [推荐系统](https://paddlepedia.readthedocs.io/en/latest/tutorials/recommendation_system/index.html)  

* **产业实践篇**  

  1. [模型压缩](https://paddlepedia.readthedocs.io/en/latest/tutorials/model_compress/index.html)
  2. [模型部署](https://paddlepedia.readthedocs.io/en/latest/tutorials/model_deployment/index.html)  

* **强化学习篇**  

  1. [强化学习](https://paddlepedia.readthedocs.io/en/latest/tutorials/reinforcement_learning/index.html)  

* <span id ='6'>**面试宝典**</span>  
  1.  [深度学习基础常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html)
  2. [卷积模型常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id2)
  3. [预训练模型常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id3)
  4. [对抗神经网络常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id4)
  5. [计算机视觉常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id5)
  6. [自然语言处理常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id6)
  7. [推荐系统常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id7)
  8.  [模型压缩常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id8)
  9.  [强化学习常见面试题](https://paddlepedia.readthedocs.io/en/latest/tutorials/interview_questions/interview_questions.html#id9)


返回[:arrow_heading_up:](#0)

-----

# 👉案例类

## <span id ='5'>飞桨应用案例集</span>

| 领域         | 产业案例                                                     | 来源                                                         | 更多内容                                                     |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **智能工业** | [厂区传统仪表统计监测](https://paddlex.readthedocs.io/zh_CN/develop/examples/meter_reader.html) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智能工业** | [新能源汽车锂电池隔膜质检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2104) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智能工业** | [天池铝材表面缺陷检测](https://paddlex.readthedocs.io/zh_CN/develop/examples/industrial_quality_inspection/README.html) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智能工业** | [安全帽检测](https://github.com/PaddleCV-FAQ/PaddleDetection-FAQ/blob/main/Lite%E9%83%A8%E7%BD%B2/yolov3_for_raspi.md) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧城市** | [高尔夫球场遥感监测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2103) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧城市** | [积雪语义分割](https://paddlex.readthedocs.io/zh_CN/develop/examples/multi-channel_remote_sensing/README.html) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧城市** | [戴口罩的人脸识别](https://aistudio.baidu.com/aistudio/projectdetail/267322?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧交通** | [车道线分割和红绿灯安全检测](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/vehicle/README_cn.md) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧交通** | [【PaddleDetection2.0专项】PP-YOLOv2](https://aistudio.baidu.com/aistudio/projectdetail/1922155?channelType=0&channel=0) | 飞桨PaddleDet                                                | [更多paddleDet案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/330600) |
| **智慧交通** | [PaddleX助力无人驾驶（基于YOLOv3的车辆检测和车道线分割）](https://aistudio.baidu.com/aistudio/projectdetail/464339?channelType=0&channel=0) | 开发者[BIT可达鸭](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧交通** | [eblite_标志物检测](https://aistudio.baidu.com/aistudio/projectdetail/596152?channelType=0&channel=0) | 开发者[TobeWell](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/59591) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧交通** | [PaddleOCR: 车牌识别](https://aistudio.baidu.com/aistudio/projectdetail/739559?channelType=0&channel=0) | 飞桨开发者[寂寞你快进去](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/180581) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧农林** | [耕地地块识别](https://mp.weixin.qq.com/s/JlDVmYlhN7sF0hpRlncDNw) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧农林** | [AI识虫](https://aistudio.baidu.com/aistudio/projectdetail/439888) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧农林** | [更快更强！ 高效快速的PP-YOLO实战演练](https://aistudio.baidu.com/aistudio/projectdetail/708923?channelType=0&channel=0) | 飞桨PaddleDet                                                | [更多paddleDet案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/330600) |
| **智慧农林** | [PaddleX快速上手-Faster RCNN目标检测](https://aistudio.baidu.com/aistudio/projectdetail/439888?channelType=0&channel=0) | 飞桨PaddleX                                                  | [更多PaddleX案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/189619) |
| **智慧农林** | [AI识虫检测分享](https://aistudio.baidu.com/aistudio/projectdetail/289616?channelType=0&channel=0) | 开发者[aaaLKgo](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/110992) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧农林** | [基于PaddleX实现森林火灾监测](https://aistudio.baidu.com/aistudio/projectdetail/1968964?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧医疗** | [医学常见中草药分类](https://aistudio.baidu.com/aistudio/projectdetail/1434738?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧医疗** | [眼疾识别](https://www.paddlepaddle.org.cn/tutorials/projectdetail/1630501) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧医疗** | [基于Paddle的肝脏CT影像分割](https://aistudio.baidu.com/aistudio/projectdetail/250994?channelType=0&channel=0) | 开发者[代码生成器](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/33061) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **智慧医疗** | [PaddleHub 肺炎CT影像分析](https://aistudio.baidu.com/aistudio/projectdetail/289819?channelType=0&channel=0) | 飞桨PaddleHub                                                | [更多PaddleHub案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/79927) |
| **智慧医疗** | [基于飞桨PGL的高致病性传染病的传播趋势预测基线系统](https://aistudio.baidu.com/aistudio/projectdetail/457185?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [人摔倒检测](https://aistudio.baidu.com/aistudio/projectdetail/2071768) | 开发者[Niki_173](https://github.com/Niki173)                 | [该开发者更多案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/474269) |
| **其他**     | [足球比赛动作定位](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [基于强化学习的飞行器仿真](https://github.com/PaddlePaddle/PARL/tree/develop/examples/tutorials/homework/lesson5/ddpg_quadrotor) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [基于ERNIE-Gram实现语义匹配](https://aistudio.baidu.com/aistudio/projectdetail/2247755) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [『NLP打卡营』实践课5：文本情感分析](https://aistudio.baidu.com/aistudio/projectdetail/1968542?channelType=0&channel=0) | 飞桨PaddleNLP                                                | [更多飞桨PaddleNLP案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995) |
| **其他**     | [『NLP经典项目集』03：利用情感分析选择年夜饭](https://aistudio.baidu.com/aistudio/projectdetail/1468469?channelType=0&channel=0) | 飞桨PaddleNLP                                                | [更多飞桨PaddleNLP案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995) |
| **其他**     | [分类任务：如何在客服对话中，识别客户情绪的好坏](https://aistudio.baidu.com/aistudio/projectdetail/121630?channelType=0&channel=0) | 开发者[中大bbking](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/34238) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [『NLP打卡营』实践课3：使用预训练模型实现快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361?channelType=0&channel=0) | 飞桨PaddleNLP                                                | [更多飞桨PaddleNLP案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995) |
| **其他**     | [发愁七夕文案？PaddleHub情话生成送给你 (文内含七夕抽奖)](https://aistudio.baidu.com/aistudio/projectdetail/746002?channelType=0&channel=0) | 飞桨PaddleHub                                                | [更多PaddleHub案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/79927) |
| **其他**     | [基于PaddleDetection的PCB瑕疵检测](https://aistudio.baidu.com/aistudio/projectdetail/2240725) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [基于百度飞桨的单/多镜头行人追踪（非官方Baseline）](https://aistudio.baidu.com/aistudio/projectdetail/1411754?channelType=0&channel=0) | 开发者[BIT可达鸭](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [PaddleLite树莓派从0到1：安全帽检测小车部署（一）](https://aistudio.baidu.com/aistudio/projectdetail/1059610?channelType=0&channel=0) | 开发者[深渊上的炕](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/90149) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [PaddleX、PP-Yolo：手把手教你训练、加密、部署目标检测模型](https://aistudio.baidu.com/aistudio/projectdetail/920753?channelType=0&channel=0) | 开发者[深渊上的炕](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/90149) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [中文语音识别](https://aistudio.baidu.com/aistudio/projectdetail/2280562) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [PaddleHub一键OCR中文识别(超轻量8.1M模型，火爆)](https://aistudio.baidu.com/aistudio/projectdetail/507159?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285?channelType=0&channel=0) | 飞桨PaddleGAN                                                | [更多PaddleGAN案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/52570) |
| **其他**     | [天下第一AI武道会-Deepfake换脸](https://aistudio.baidu.com/aistudio/projectdetail/1189026?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [飞桨创意之星 宋代诗人念诗的秘密——PaddleGAN实现精准唇形合成](https://aistudio.baidu.com/aistudio/projectdetail/1463208?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [通过OCR实现验证码识别](https://aistudio.baidu.com/aistudio/projectdetail/1100507?channelType=0&channel=0) | 飞桨官方                                                     | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [PaddleHub一键OCR中文识别（超轻量8.1M模型，火爆）](https://aistudio.baidu.com/aistudio/projectdetail/507159?channelType=0&channel=0) | 飞桨PaddleHub                                                | [更多PaddleHub案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/79927) |
| **其他**     | [全流程，从零搞懂基于PaddlePaddle的图像分割](https://aistudio.baidu.com/aistudio/projectdetail/1674328?channelType=0&channel=0) | 开发者[nanting03](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/129509) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [负荷预测0.1](https://aistudio.baidu.com/aistudio/projectdetail/2183242?channelType=0&channel=0) | 开发者[gaomaosheng0](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/29822) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | [AI 实现皮影戏，传承正在消失的艺术](https://aistudio.baidu.com/aistudio/projectdetail/764130?channelType=0&channel=0) | 开发者[Zohar](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/331031) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **其他**     | 『[深度学习7日打卡营』人脸关键点检测](https://aistudio.baidu.com/aistudio/projectdetail/1487972?channelType=0&channel=0) | 开发者[TC.Long](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/157251) | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase) |
| **强化学习** | [DDPG算法应用于股票量化交易](https://github.com/PaddlePaddle/awesome-DeepLearning/tree/master/examples/DDPG%20for%20Stock%20Trading) | 开发者                                                       | [更多飞桨案例](https://www.paddlepaddle.org.cn/customercase?fr=paddleEdu_github) |

## <span id ='5'>飞桨学术案例集</span>

| 技术方向     | 学术案例                                                     | 来源                                                         | 更多内容                                                     |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 机器学习     | [鸢尾花分类](https://aistudio.baidu.com/aistudio/projectdetail/78918?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 前馈神经网络 | [波士顿房价预测](https://aistudio.baidu.com/aistudio/projectdetail/79112?channelType=0&channel=0) | [开发者AIStudioHelper](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/47528) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [手写数字识别](https://aistudio.baidu.com/aistudio/projectdetail/325575?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [猫狗分类](https://aistudio.baidu.com/aistudio/projectdetail/78960?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [图像分类网络VGG在多表情识别任务中的应用](https://aistudio.baidu.com/aistudio/projectdetail/2369842) | [开发者之雍Jerry](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/530098) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [图像分类-ResNet](https://aistudio.baidu.com/aistudio/projectdetail/56779?channelType=0&channel=0) | [开发者笨笨](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/39) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [用PaddlePaddle实现图像分类-SE_ResNeXt](https://aistudio.baidu.com/aistudio/projectdetail/169410?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [深入理解图像分类中的Transformer-Vit,DeiT](https://aistudio.baidu.com/aistudio/projectdetail/2293050) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [Swin   Transformer](https://aistudio.baidu.com/aistudio/projectdetail/2280436) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分类     | [小样本学习(Few-Shot   Learning)](https://aistudio.baidu.com/aistudio/projectdetail/2342018?channelType=0&channel=0) | [开发者DeepGeGe](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/746341) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分割     | [经典实例分割模型Mask   RCNN](https://aistudio.baidu.com/aistudio/projectdetail/122273?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分割     | [PaddleSeg_DeepLabv3+](https://aistudio.baidu.com/aistudio/projectdetail/226703?channelType=0&channel=0) | [飞桨PaddleSeg](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/96056) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像分割     | [基于PaddlePaddle的语义分割DeepLabV3+实现](https://aistudio.baidu.com/aistudio/projectdetail/124366?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像检测     | [深度学习进阶-目标检测](https://aistudio.baidu.com/aistudio/projectdetail/78972?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像检测     | [一文详解yolov3目标检测算法](https://aistudio.baidu.com/aistudio/projectdetail/2240328) | [开发者AIStudio96069](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/96069) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 图像检测     | [CV领域的Transformer模型DETR在目标检测任务中的应用](https://aistudio.baidu.com/aistudio/projectdetail/2290729) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 视频分类     | [TSN视频分类](https://aistudio.baidu.com/aistudio/projectdetail/2280460) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 视频分类     | [Paddle2.1实现视频理解经典模型  — TSM](https://aistudio.baidu.com/aistudio/projectdetail/2311166?contributionType=1) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 视频分类     | [基于Attention和Bi-LSTM实现视频分类](https://aistudio.baidu.com/aistudio/projectdetail/2313514) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 视频分类     | [CV领域的Transformer模型TimeSformer实视频理解](https://aistudio.baidu.com/aistudio/projectdetail/2291410) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| GAN          | [一文搞懂生成对抗网络之经典GAN（动态图、VisualDL2.0）](https://aistudio.baidu.com/aistudio/projectdetail/551962?channelType=0&channel=0) | [开发者FutureSI](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| GAN          | [基于PaddlePaddle的StarGAN,AttGAN,STGAN算法](https://aistudio.baidu.com/aistudio/projectdetail/169439?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| OCR          | [文字识别-CRNN](https://aistudio.baidu.com/aistudio/projectdetail/190025?channelType=0&channel=0) | [开发者哦吼](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/34706) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [基于ERNIE实现9项GLUE任务](https://aistudio.baidu.com/aistudio/projectdetail/2345396) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [NLP领域的XLNet模型在情感分析中的应用](https://aistudio.baidu.com/aistudio/projectdetail/2333184) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [NLP领域中的ERNIE模型在阅读理解中的应用](https://aistudio.baidu.com/aistudio/projectdetail/2333137) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [NLP领域的ELECTRA在符号预测上的应用](https://aistudio.baidu.com/aistudio/projectdetail/2311092) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [NLP领域的Transformer在机器翻译上的应用](https://aistudio.baidu.com/aistudio/projectdetail/2311016) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [【Paddle打比赛】讯飞赛题—中文问题相似度挑战赛0.9+Baseline](https://aistudio.baidu.com/aistudio/projectdetail/2271498) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| NLP          | [用PaddlePaddle实现BERT](https://aistudio.baidu.com/aistudio/projectdetail/122282?channelType=0&channel=0) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 多模态       | [【Paddle   CLIP】你写啥他画啥，一个专属于你的小画家](https://aistudio.baidu.com/aistudio/projectdetail/2332016?channelType=0&channel=0) | [PaddleFleet](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/940489) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 强化学习     | [从代码到论文理解并复现MADDPG算法(PARL)](https://aistudio.baidu.com/aistudio/projectdetail/637951?channelType=0&channel=0) | [开发者Mr.郑先生_](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/147378) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 推荐         | [基于   DeepFM 模型的点击率预估](https://github.com/PaddlePaddle/awesome-DeepLearning/tree/master/examples/DeepFM for CTR Prediction) | [PaddleEdu](https://github.com/PaddlePaddle/awesome-DeepLearning) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 推荐         | [基于DSSM的电影推荐](https://aistudio.baidu.com/aistudio/projectdetail/2324144) | [AIStudio官方](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/7) | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |
| 知识蒸馏     | [基于CIFAR100的SSLD蒸馏实验](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/docs/zh_CN/advanced_tutorials/distillation/distillation.md) | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)     | [更多飞桨案例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/908086) |

返回[:arrow_heading_up:](#0)

----

# 👉竞赛类

| 领域     | 竞赛案例                                                     | 介绍                                                         |
| -------- | ------------------------------------------------------------ | ---- |
| 机器学习 |                                                              |                                                              |
| NLP      | [【Paddle打比赛】讯飞赛题—中文问题相似度挑战赛0.9+Baseline](https://aistudio.baidu.com/aistudio/projectdetail/2271498) | 中文问题相似度挑战赛paddle版本Baseline，基于paddlenlp通过预训练模型的微调完成问题相似度评定任务 |
| NLP      | [基于PaddleHub的疫情期间网民情绪识别](https://aistudio.baidu.com/aistudio/projectdetail/294224?channelType=0&channel=0) | 本项目为疫情期间网民情绪识别比赛的解决方案。使用了PaddleHub和ERNIE实现对疫情期间微博文本的情绪识别。 |
| 语音     |                                                              |                                                              |
| CV       | [中文场景文字识别挑战赛baseline](https://aistudio.baidu.com/aistudio/projectdetail/229728?channelType=0&channel=0) | 中文场景文字识别挑战赛的baseline项目, 用于参赛选手借鉴参考   |
| CV       | [2020 CCF BDCI: 遥感影像地块分割baseline](https://aistudio.baidu.com/aistudio/projectdetail/1090790?channelType=0&channel=0) | 2020 CCF BDCI: 遥感影像地块分割的baseline模型库，包括baseline模型的训练方法和比赛的评测脚本。 |
| CV       | [第三届中国AI+创新创业大赛：半监督学习目标定位竞赛第1名方案](https://aistudio.baidu.com/aistudio/projectdetail/2210815)| 半监督学习目标定位竞赛第一名方案分享 A榜得分0.81425 B榜得分0.80428 |
| 推荐     |                                                              |                                                              |
| 强化学习 |                                                              ||

返回​[:arrow_heading_up:](#0)


# 👉汇总

## <span id='fj'>飞桨各产品学习资料汇总</span>

| 产品                             | 视频课程                                                     | 学习文档 |
| -------------------------------- | ------------------------------------------------------------ | -------- |
| PaddleGAN                        | [生成对抗网络七日打卡营](https://aistudio.baidu.com/aistudio/course/introduce/16651) |          |
| PaddleOCR                        | [OCR自动标注小工具讲解](https://www.bilibili.com/video/BV1uX4y1K7PW)、[3.5M超轻量实用OCR模型解读](https://www.bilibili.com/video/BV1p54y1y7CM)、[OCR应用与部署实战](https://www.bilibili.com/video/BV1Zz4y1C7MW) |          |
| PaddleClas                       | [PaddleClas系列直播课](https://aistudio.baidu.com/aistudio/course/introduce/24519) |          |
| PaddleDetection                  | [目标检测7日打卡营](https://aistudio.baidu.com/aistudio/course/introduce/1617) |          |
| PaddleX                          | [PaddleX实例分割任务详解](https://www.bilibili.com/video/BV1M44y1r7s6)、[PaddleX目标检测任务详解](https://www.bilibili.com/video/BV1HB4y1A73b)、[PaddleX语义分割任务详解](https://www.bilibili.com/video/BV1qQ4y1Z7co)、[PaddleX图像分类任务详解](https://www.bilibili.com/video/BV1nK411F7J9)、[PaddleX客户端操作指南](https://www.bilibili.com/video/BV1bz4y1C7wr)、[飞桨全流程开发工具PaddleX](https://www.bilibili.com/video/BV17i4y1b7TZ) |          |
| <span id ='hub'>PaddleHub</span> | [手把手教你转换PaddleHub模型教程](https://www.bilibili.com/video/BV1YK411V71d) |          |
| <span id = 'vdl'>VDL</span>      | [可视化分析工具助力AI算法快速开发](https://www.bilibili.com/video/BV1uy4y137iH)、[深度学习算法可视化调优实战演示](https://www.bilibili.com/video/BV1iD4y1o7Pf) |          |
| 高层API                          | [高层API助你快速上手深度学习](https://aistudio.baidu.com/aistudio/course/introduce/6771) |          |
| <span id='nlp'>PaddleNLP</span>  | [基于深度学习的自然语言处理](https://www.bilibili.com/video/BV1fB4y1M7A3) |          |

返回​[:arrow_heading_up:](#0)


# 三、技术交流

非常感谢您使用本项目。您在使用过程中有任何建议或意见，可以在 **[Issue](https://github.com/PaddlePaddle/tutorials/issues)** 上反馈给我们，也可以通过扫描下方的二维码联系我们，飞桨的开发人员非常高兴能够帮助到您，并与您进行更深入的交流和技术探讨。

<center><img src="https://github.com/ZhangHandi/images-for-paddledocs/blob/main/images/readme/qr_code.png?raw=true"/></center><br></br>

# 四、许可证书

本项目的发布受[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.txt)许可认证。

# 五、贡献内容

本项目的不断成熟离不开各位开发者的贡献，如果您对深度学习知识分享感兴趣，非常欢迎您能贡献给我们，让更多的开发者受益。

本项目欢迎任何贡献和建议，大多数贡献都需要你同意参与者许可协议（CLA）来声明你有权并实际上授权我们可以使用你的贡献。

### 代码贡献规范

> pip install pre-commit
>
> pre-commit install

添加修改的代码后，对修改的文件进行代码规范，pre-commit 会自动调整代码格式，执行一次即可，后续commit不需要再执行。提交pr流程，详见：[awesome-DeepLearning 提交 pull request 流程](./examples/awesome-DeepLearning_pr_procedure.md)。

### 贡献者

以下是awesome-DeepLearning贡献者列表： [yang zhou](https://youngzhou1999.github.io/cv/)，[Niki_173](https://github.com/Niki173)，[Twelveeee](https://github.com/Twelveeee)，[buriedms](https://github.com/buriedms)，[AqourAreA](https://github.com/AqourAreA)，[zhangjin12138](https://github.com/zhangjin12138)，[rerny](https://github.com/rerny)，[LiuCongNLP](https://www.zhihu.com/people/LiuCongNLP)

