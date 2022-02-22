# 论文简介
![](https://ai-studio-static-online.cdn.bcebos.com/a7bc44afcfa14e13bcd0ed4045593ad8433aa7a081e74f89b4154ba1fa6d27ae)  
  论文作者提出一种新的、全监督语义分割训练范式，可应用于语义分割的训练中，能够显著提高语义分割的效果。论文在cityscapes验证集上进行实验，HRNet_W48的mIou为81.0%，本次复现赛要求复现的精度为82.2%，本项目复现结果为82.47%。
## 论文核心思想
  作者提出一种新的、全监督语义分割训练范式，利用跨图像的像素-像素之间的关系，来学习一个更好的特征空间。如上图所示，（a）和（b）是训练图片及其对应的标签，传统的语义分割训练时忽略了不同图片之间的上下文信息，而本篇论文提出的跨图像像素对比学习，利用不同图片之间的像素关系，使得特征空间中同种类别的像素相似性变高、不同类别的像素相似度变低（如图d），从而得到一个更好的结构化的特征空间（如图e）。
## 网络结构
![](https://ai-studio-static-online.cdn.bcebos.com/0e7d7d21221f4ed58688cccf39106dc115cb85965cf9448189c686a1ddb1b4bb)  
上图为网络结构，fFCN为backbone模块，fSEG为语义分割head模块，从fSEG平行引出fPROJ模块，fPROJ用来进行对比训练，fSEG训练方法与传统方法相同。
## 损失函数
![](https://ai-studio-static-online.cdn.bcebos.com/3dc4076cf08b46d5b74551491068602b6dabc38a8f44463692026cffae4d2d3e)  
损失函数如上图，由2部分构成，交叉熵损失和对比损失组成，其中对比损失为本篇论文的核心。  

![](https://ai-studio-static-online.cdn.bcebos.com/5b58c4ca0bb24ae1b23fa1e1b713051828bbd0ce4461466baf6c715f81a1ea06)  
交叉熵损失如上图所示，此处不做介绍。  

![](https://ai-studio-static-online.cdn.bcebos.com/7f7cfacb399142a6b0afe127f62f5a2613de0dc069674f86a46f92e0be51ca37)  
上图为对比损失，亦是本篇论文的核心。其中i是真实标签为c的特征向量，i+为正样本像素特征，i-为负样本像素特征，由上式可以看出，通过像素-像素对比学习，在特征空间上同一类别的像素拉近，不同类别的像素原理，从而使得不同类别的像素特征空间能够更好的可区分。  

## 对比损失的anchor采样方法
预测错误的像素被认为是hard anchor，预测正确的像素被认为是easy anchor，在对比度损失计算过程中，一半的anchor是hard anchor，另一半是easy anchor。  

## 实验
![](https://ai-studio-static-online.cdn.bcebos.com/bce32da16bb347bd900ea39d4d03538568d14088286044e8999777e95b96193f)  
上左图为像素交叉熵损失的特征可视图，上右图是对比损失的特征可视化图，可以看出，使用了像素对比损失的语义分割模型特征空间更具结构化。


# 项目介绍
本项目为第四届百度论文复现赛Exploring Cross-Image Pixel Contrast for Semantic Segmentation复现结果，本项目基于PaddleSeg实现，并复现了论文实验结果，非常感谢百度提供的比赛平台以及GPU资源。  

## 复现环境
硬件： Tesla V100 * 4  
框架： paddlepaddle==2.1.2  

## 参数调试
1、project dimension：此参数不影响最后的模型大小，可以适度调整来获得好的效果；  
2、迭代次数：40k次迭代时mIou为81.8%，不满足验收标准，60k次迭代mIou为82.47%，可以适当提高迭代次数；  
3、训练时长：Tesla V100 * 4， batch_size为2，60k次迭代，训练总时长为13.5h。  

## 复现经验
1、使用paddleseg语义分割框架可以有效的减少复现的代码量，提高复现速度；  
2、论文提供的源码使用的框架是pytorch，需要注意不同框架模型转换的细节（尤其是部分函数的不同）。  

## 项目结果

Method|Environment|mIou|Step|Batch_size|Dataset
:--:|:--:|:--:|:--:|:--:|:--:
HRNet_W48_Contrast|Tesla V-100 $\times$ 4 |82.47|60k|2|CityScapes


## 项目实现思路
本项目基于PaddleSeg开发，由于PaddleSeg已经包含了数据处理，验证等多项功能，本篇论文的复现仅需要实现HRNet_W48 Head和Contrast loss部分。


## 项目运行实践
1、按照以下步骤可运行本项目。  
2、由于HRNet_W48参数较大，无法上传至Aistudio，可从github的百度云连接下载训练好的权重参数。


```python
# step 1: git clone, 本项目已clone过，跳过此步骤
!git clone https://hub.fastgit.org/justld/contrast_seg_paddle.git
```


```python
# step 2: 解压数据集
%cd /home/aistudio/data/data64550
!tar -xf cityscapes.tar
%cd /home/aistudio
```


```python
# step 3: 训练
# 注意：把configs目录下的HRNet_W48_cityscapes_1024x512_60k中数据集目录更换为自己的数据集目录，如：dataset_root: /home/aistudio/data/data64550/cityscapes
%cd /home/aistudio/contrast_seg_paddle
!python train.py --config configs/HRNet_W48_cityscapes_1024x512_60k.yml  --do_eval --use_vdl --log_iter 100 --save_interval 1000 --save_dir output
```


```python
# step 4: 验证
# 注意：由于训练好的参数文件太大，请移步github（https://hub.fastgit.org/justld/contrast_seg_paddle.git）从百度云下载权重文件，并把model_path更换为自己的参数路径
%cd /home/aistudio/contrast_seg_paddle
!python val.py --config configs/HRNet_W48_cityscapes_1024x512_60k.yml --model_path output/best_model/model.pdparams

```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
