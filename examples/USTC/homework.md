【作业】
1.	使用CIFAR10数据集，基于EffNet网络实现图像分类。
2.	使用CIFAR10数据集，基于DarkNet网络实现图像分类。
3.	在眼疾识别数据集上训练SENet网络。
4.	在眼疾识别数据集上训练SqueezeNet网络。
5.	在眼疾识别数据集上训练DPN网络。
6.	使用THUCNews标题数据集，基于textcnn网络实现文本分类。
7.	基于LSTM网络训练一个语言模型，并尝试用于下一个词预测任务进行效果验证。
8.	使用LCQMC数据集，基于LSTM网络训练一个文本匹配模型。
9.	使用ChnSentiCorp数据集，基于GRU网络完成情感分析模型。
10.	手动实现LSTM模型，并尝试利用IMDB数据集进行情感分析。

参考链接：https://aistudio.baidu.com/aistudio/education/group/info/1297
https://github.com/PaddlePaddle/PaddleClas


要求:
实训作业分为理论讲解和代码实现两部分，模型理论要求对实训作业中使用的模型进行详细的解释，图文并茂为佳。 
代码实现部分书写流程如下:
1.	实验设计逻辑:解释任务，说明实验设计逻辑 
2.	数据处理:解释数据集，处理数据为模型输入格式 
3.	模型设计:根据任务设计模型，需要给出模型设计图 
4.	训练配置:定义模型训练的超参数，模型实例化，指定训练的cpu或gpu资 源，定义优化器等等 
5.	模型训练与评估:训练模型，在训练过程中，根据开发集适时打印结果 
6.	模型推理:设计一个接口函数，通过这个接口函数能够方便地对任意一个样本进行实时预测 

特殊任务如强化学习可以省略数据处理和模型推理部分。书写流程可参考：https://aistudio.baidu.com/aistudio/projectdetail/2023570

作业命名及pr commit要以作业内容进行命名，需用英文命名，不可以使用学号/姓名或xxx homework等形式。命名方式举例：image classification with VGG model on CIFAR-10 dataset.

提交格式:
使用 aistudio 实训平台进行作业实现，在 github repo 中以提pr的形式提交 aistudio 链接（注意:作业需要公开才能看到)，注意保留训练结果和日志；提交 python 版本可以获得额外 加分，python 版本需要将代码按模块拆分，并提交详细的 readme(包括模型简介、数据 准备、模型训练、模型测试及测试结果、模型推理和参考论文)，可参考: https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/tsm.md

作业提交地址:
https://github.com/PaddlePaddle/awesome-DeepLearning/tree/master/examples

提交pr流程:
https://github.com/PaddlePaddle/awesome-DeepLearning/blob/master/examples/awesome-DeepLearning_pr_procedure.md


