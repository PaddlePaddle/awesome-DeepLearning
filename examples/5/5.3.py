#!/usr/bin/env python
# coding: utf-8

# ## 注意
# 
# 本项目代码包含多个文件, Fork并使用GPU环境来运行后, 才能看到项目完整代码, 并正确运行:
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/767f625548714f03b105b6ccb3aa78df9080e38d329e445380f505ddec6c7042" width="40%" height="40%">
# 
# <br>
# 
# <br>
# 
# 并请检查相关参数设置, 例如use_gpu, fluid.CUDAPlace(0)等处是否设置正确. 

# 
# 
# # 语言模型
# 
# # 简介
# 
# ## 1. 任务说明
# 本文主要介绍基于lstm的语言的模型的实现，给定一个输入词序列（中文分词、英文tokenize），计算其ppl（语言模型困惑度，用户表示句子的流利程度），基于循环神经网络语言模型的介绍可以[参阅论文](https://arxiv.org/abs/1409.2329)。相对于传统的方法，基于循环神经网络的方法能够更好的解决稀疏词的问题。
# 
# ## 2. 效果说明
# 在small meidum large三个不同配置情况的ppl对比：
# 
# |  small config  |    train    |   valid    |    test      |
# | :------------- | :---------: | :--------: | :----------: |
# |     paddle     |    40.962   |  118.111   |   112.617    |
# |   tensorflow   |    40.492   |  118.329   |   113.788    |
# 
# |  medium config |    train    |   valid    |    test      |
# | :------------- | :---------: | :--------: | :----------: |
# |     paddle     |    45.620   |  87.398    |    83.682    |
# |   tensorflow   |    45.594   |  87.363    |    84.015    |
# 
# |  large config  |    train    |   valid    |    test      |
# | :------------- | :---------: | :--------: | :----------: |
# |     paddle     |    37.221   |  82.358    |    78.137    |
# |   tensorflow   |    38.342   |  82.311    |    78.121    |
# 
# ## 3. 数据集
# 此任务的数据集合是采用ptb dataset，下载地址为: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# 
# 
# # 快速开始
# 
# ## 1. 开始第一次模型调用
# 
# 
# ### 训练或fine-tune
# 任务训练启动命令如下：
# ```
# !python train.py --use_gpu True --data_path data/data11325/simple-examples/data --model_type small --rnn_model basic_lstm
# ```
# - 需要指定数据的目录，默认训练文件名为 ptb.train.txt，可用--train_file指定；默认验证文件名为 ptb.valid.txt，可用--eval_file指定；默认测试文件名为 ptb.test.txt，可用--test_file指定
# - 模型的大小(默认为small，用户可以选择medium， 或者large)
# - 模型的类型（默认为static，可选项static|padding|cudnn|basic_lstm）
# - batch大小默认和模型大小有关，可以通过--batch_size指定
# - 训练轮数默认和模型大小有关，可以通过--max_epoch指定
# - 默认将模型保存在当前目录的models目录下
# 
# # 进阶使用
# ## 1. 任务定义与建模
# 此任务目的是给定一个输入的词序列，预测下一个词出现的概率。
# 
# ## 2. 模型原理介绍
# 此任务采用了序列任务常用的rnn网络，实现了一个两层的lstm网络，然后lstm的结果去预测下一个词出现的概率。计算的每一个概率和实际下一个词的交叉熵，然后求和，做e的次幂，得到困惑度ppl。当前计算方式和句子的长度有关，仍需要继续优化。
# 
# 由于数据的特殊性，每一个batch的last hidden和last cell会被作为下一个batch 的init hidden 和 init cell，数据的特殊性下节会介绍。
# 
# 
# ## 3. 数据格式说明
# 此任务的数据格式比较简单，每一行为一个已经分好词（英文的tokenize）的词序列。
# 
# 目前的句子示例如下图所示:
# ```
# aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo wachter
# pierre <unk> N years old will join the board as a nonexecutive director nov. N
# mr. <unk> is chairman of <unk> n.v. the dutch publishing group
# ```
# 特殊说明：ptb的数据比较特殊，ptb的数据来源于一些文章，相邻的句子可能来源于一个段落或者相邻的段落，ptb 数据不能做shuffle
# 
# 
# 
# ## 4. 目录结构
# 
# ```text
# .
# ├── train.py             # 训练代码
# ├── reader.py            # 数据读取
# ├── args.py              # 参数读取
# ├── config.py              # 训练配置
# ├── data                # 数据下载
# ├── language_model.py  		  # 模型定义文件
# ```
# 
# ## 5. 如何组建自己的模型
# + **自定义数据：** 关于数据，如果可以把自己的数据先进行分词（或者tokenize），然后放入到data目录下，并修改reader.py中文件的名称，如果句子之间没有关联，用户可以将`train.py`中更新的代码注释掉。
#     ```
#     init_hidden = np.array(fetch_outs[1])
#     init_cell = np.array(fetch_outs[2])
#     ```
# 
# + **网络结构更改：** 网络只实现了基于lstm的语言模型，用户可以自己的需求更换为gru或者self等网络结构，这些实现都是在language_model.py 中定义

# In[1]:


# 解压数据集
get_ipython().system('cd data/data11325 && unzip -qo simple-examples.zip')


# In[2]:


# 运行训练，使用GPU，并且使用小模型
# 训练轮数也限制到3轮，以避免日志过多
# 最终可以通过提高训练轮数达到比较好的效果
get_ipython().system('echo "training"')
get_ipython().system('python train.py --use_gpu True --data_path data/data11325/simple-examples/data --model_type small --rnn_model basic_lstm --max_epoch=3')


# In[12]:


# 加模型进行预测
get_ipython().system('python infer.py --rnn_model basic_lstm')


# In[ ]:




