# 论文复现：PointRend: Image Segmentation as Rendering(CVPR2020)

## 一、简介

本项目利用百度的paddlepaddle框架对CVPR2020论文PointRend进行了复现，在Cityscapes数据集上进行了对于SemanticFPN+PointRend的语义分割实验。

该论文是计算机视觉顶级会议CVPR2020的会议论文,论文提出了一种新的方法，可以对物体和场景进行有效的高质量图像分割。

具体来讲，核心idea采用了将图像分割作为渲染问题的独特视角。从这个角度出发，论文提出了PointRend（基于点的渲染）神经网络模块：该模块基于迭代细分算法在自适应选择的位置执行基于点的分割预测。通过在现有最新模型的基础上构建，PointRend可以灵活地应用于实例和语义分割任务。
定性地结果分析表明，PointRend在先前方法过度平滑的区域中输出清晰的对象边界。定量来看，无论是实例还是语义分割，PointRend都在COCO和Cityscapes上产生了有效的性能提升。PointRend模块作为可附加的模块提升现有网络的分割结果。

**PointRend With Seg Architecture:**
- PointRend method:
<center><img src="https://img-blog.csdnimg.cn/20200223162103205.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM0Njg2MTU4,size_16,color_FFFFFF,t_70"， height=40%, width=40%></center>

- PointRend result
<center><img src="https://img-blog.csdnimg.cn/20200223163520107.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM0Njg2MTU4,size_16,color_FFFFFF,t_70"， height=40%, width=40%></center>


**通俗易懂的解读博客推荐：**

> [PointRend](https://blog.csdn.net/sinat_34686158/article/details/104457405)

> [2020CVPR解读之何恺明新作PointRend：将图像分割视作渲染问题，显著提升语义/实例分割性能](2020CVPR解读之何恺明新作PointRend：将图像分割视作渲染问题，显著提升语义/实例分割性能)


**论文地址:** [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)


## 二、复现精度

| Model                   | mIOU |
| ----------------------- | -------- |
| SemanticFPN+PointRend(原文Pytorch)     | 78.5     |
| SemanticFPN+PointRend(本项目Paddle) | 78.78  |

(预训练模型（四卡训练）已经放置在output目录下，可按下方提供的评估指令进行评估)

## 三、数据集

使用的数据集为：[Cityscapes](https://www.cityscapes-dataset.com/)

- 数据集大小：19个类别的密集像素标注，5000张1024*2048大小的高质量像素级注释图像/20000个弱注释帧
  - 训练集：2975个图像
  - 验证集：500个图像
  - 测试集：1525个图像

数据集应有的结构:
```
data/
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── leftImg8bit
│   │   ├── test
│   │   │   ├── berlin
│   │   │   ├── ...
│   │   │   └── munich
│   │   ├── train
│   │   │   ├── aachen
│   │   │   ├── ...
│   │   │   └── zurich
│   │   └── val
│   │       ├── frankfurt
│   │       ├── lindau
│   │       └── munster
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt

```

.txt是利用Paddleseg提供的数据集处理工具生成，其风格如下:
```leftImg8bit/test/mainz/mainz_000001_036412_leftImg8bit.png,gtFine/test/mainz/mainz_000001_036412_gtFine_labelTrainIds.png```

利用PaddleSeg's create_dataset_list.py(需要先克隆[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)):

```
python PaddleSeg/tools/create_dataset_list.py ./data/cityscapes/ --type cityscapes --separator ","

```
当然，需要首先生成xxxxx_gtFine_labelTrainIds.png标签。这个需要利用cityscapes提供的工具生成,具体使用方法这里不作介绍，请查阅[Cityscapes](https://www.cityscapes-dataset.com/)

**Note**：**这里的操作下面的notebook都提供了一键操作～可以直接按步骤省心运行～**

# 准备数据集



```python
# 准备数据集
# 查看当前工作目录
!pwd
# 查看工作区文件夹
!tree -d work/
# 查看数据文件夹
!tree -d data/
```

    /home/aistudio
    work/

    0 directories
    data/
    ├── data107804
    └── data48855

    2 directories



```python
# 创建cityscape文件夹
!mkdir data/cityscapes/
```


```python

# 解压数据集中的gtFine
!unzip -nq -d data/gtFine/ data/data48855/gtFine_train.zip
!unzip -nq -d data/gtFine/ data/data48855/gtFine_val.zip
!unzip -nq -d data/gtFine/ data/data48855/gtFine_test.zip
!mv data/gtFine/ data/cityscapes/
```


```python
# 解压数据集中的leftImg8bit
!unzip -nq -d data/leftImg8bit/ data/data48855/leftImg8bit_train.zip
!unzip -nq -d data/leftImg8bit/ data/data48855/leftImg8bit_val.zip
!unzip -nq -d data/leftImg8bit/ data/data48855/leftImg8bit_test.zip
!mv data/leftImg8bit/ data/cityscapes/
```


```python
# 查看工作区文件夹
# !tree -d work/
# 查看数据文件夹
!tree  -d data/cityscapes/
```


```python
!python  PaddleSeg/tools/create_dataset_list.py /home/aistudio/data/cityscapes/ --type cityscapes --separator ","
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.

# 训练
*已经验证

**注意**：1、项目是在脚本任务中完成，PointRend的配置文件是4卡的配置。若要在当前notebook中尝试复现，需要修改batchsize和学习率和iter等参数。建议学习率线性修改。
2、STDCSeg的配置文件是单卡的配置，要4卡训练也需要进行对应修改batchsize/4。

## 1、在终端执行训练

1、首先cd到Paddleseg文件夹下.

2、**可能需要在配置文件中修改数据集路径**

3、训练：

**可能报错需要修改num_workers=0。否则修改train函数Dataloader中use_shared_memory=False(请自行查阅paddle说明文档).**

* PointRend


```
python train.py --config configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml --num_workers 8 --use_vdl --do_eval --save_interval 500 --save_dir pointrend_resnet101_os8_cityscapes_1024×512_80k
```


* Stdcseg

```
python train.py --config configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml --num_workers 8 --use_vdl --do_eval --save_interval 500 --save_dir stdc2_seg_cityscapes_1024x512_80k
```


## 2、在notebook执行训练


```python
# 在notebook执行训练
!python Paddleseg/train.py --config Paddleseg/configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml --num_workers 0 --use_vdl --do_eval --save_interval 500 --save_dir pointrend_resnet101_os8_cityscapes_1024×512_80k
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-09-26 20:42:08 [INFO]
    ------------Environment Information-------------
    platform: Linux-4.4.0-166-generic-x86_64-with-debian-stretch-sid
    Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0]
    Paddle compiled with cuda: False
    GCC: gcc (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
    PaddlePaddle: 2.1.2
    OpenCV: 4.1.1
    ------------------------------------------------
    Traceback (most recent call last):
      File "Paddleseg//train.py", line 190, in <module>
        main(args)
      File "Paddleseg//train.py", line 151, in main
        train_dataset = cfg.train_dataset
      File "/home/aistudio/Paddleseg/paddleseg/cvlibs/config.py", line 343, in train_dataset
        return self._load_object(_train_dataset)
      File "/home/aistudio/Paddleseg/paddleseg/cvlibs/config.py", line 384, in _load_object
        return component(**params)
      File "/home/aistudio/Paddleseg/paddleseg/datasets/cityscapes.py", line 75, in __init__
        "The dataset is not Found or the folder structure is nonconfoumance."
    ValueError: The dataset is not Found or the folder structure is nonconfoumance.


# 评估
## 1、在notebook执行评估



```python
!python Paddleseg/val.py --config  Paddleseg/configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml  --model_path /home/aistudio/output/pointrend.pdparams
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Traceback (most recent call last):
      File "Paddleseg/val.py", line 171, in <module>
        main(args)
      File "Paddleseg/val.py", line 143, in main
        val_dataset = cfg.val_dataset
      File "/home/aistudio/Paddleseg/paddleseg/cvlibs/config.py", line 350, in val_dataset
        return self._load_object(_val_dataset)
      File "/home/aistudio/Paddleseg/paddleseg/cvlibs/config.py", line 384, in _load_object
        return component(**params)
      File "/home/aistudio/Paddleseg/paddleseg/datasets/cityscapes.py", line 75, in __init__
        "The dataset is not Found or the folder structure is nonconfoumance."
    ValueError: The dataset is not Found or the folder structure is nonconfoumance.


# 评估

*已经验证

***Note**: 注意模型路径设置

## 2、在终端执行评估：

1、首先cd到Paddleseg文件夹下.

## PointRend
* 单尺度评估

```
python val.py --config configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml  --model_path /home/aistudio/output/pointrend.pdparams
```
* 翻转评估

```
python val.py --config configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml  --model_path /home/aistudio/output/pointrend.pdparams --aug_eval --flip_horizontal
```


* 多尺度评估
```
python val.py --config configs/pointrend/pointrend_resnet101_os8_cityscapes_1024×512_80k.yml  --model_path /home/aistudio/output/pointrend.pdparams --aug_eval --scales 0.75 1.0 1.25
```

## Stdcseg

* 单尺度评估

```
python val.py --config configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml  --model_path /home/aistudio/output/model.pdparams
```
* 翻转评估

```
python val.py --config configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml  --model_path /home/aistudio/output/model.pdparams --aug_eval --flip_horizontal
```


* 多尺度评估
```
python val.py --config configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml  --model_path /home/aistudio/output/model.pdparams --aug_eval --scales 0.75 1.0 1.25
```

* m50

```
python val.py --config configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml  --model_path /home/aistudio/output/model.pdparams --aug_eval --scales 0.5
```

* m75

```
python val.py --config configs/stdcseg/stdc2_seg_cityscapes_1024x512_80k.yml  --model_path /home/aistudio/output/model.pdparams --aug_eval --scales 0.75
```

# 关于项目和论文

--> [<font face=宋体 size="3">论文链接</font>](https://arxiv.org/abs/1912.08193) </br>
--> [<font face=宋体 size="3">脚本项目</font>](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2298566) </br>
--> [<font face=宋体 size="3">参考 pointhead in mmseg</font>](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/point_head.py) </br>
--> [<font face=宋体 size="3">参考 detectron2 in mmseg</font>](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend) </br>
--> [<font face=宋体 size="3">复现 github 地址</font>](https://github.com/CuberrChen/PointRend-Paddle)
