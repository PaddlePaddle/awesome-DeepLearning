## EfficientDet介绍

**EfficientDet具有参数少，推理快，精度高的特点。基于one-stage检测方法，EfficientNet做backbone，亮点之处在于提出了BiFPN与compound scale方法**

**EfficientDet由EfficientNet Backbone, BiFPN layer, Class prediction net和 Box prediction net 四部分组成。**

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/50051adf762b43729022ddddf3839f6a651a6d8607564856b6a99e55a372b285" width="800"/></center>


BiFPN是EfficientDet的核心, 全称是"bidirectional feature network ", 也就是加权双向特征网络, 可轻松快速地进行多尺度特征融合

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/516a05c742cb4b82a201e36f8042a33207a9e09d04f9473381fd930de0acc1e6" width="800"/></center>

 - 原始FPN，只有从上到下的特征融合，只有一个方向的信息流传递

 - PANet有从上到下和从下到上的特征融合

 - NAS-FPN，搜索上需要很大资源，并且缺少可解释性和可修改性

 - BiFPN去掉了没有特征融合的只有一个输入的节点，这样做是假设只有一个输入的节点相对不太重要；在相同 level 的原始输入和输出节点之间连了一条边，假设是能融合更多特征；将该层重复多次，更更层次地融合特征

## 数据文件准备

数据集已挂载至aistudio项目中，如果需要本地训练可以从这里下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/103218)

工程目录大致如下，可根据实际情况修改
```
home/aistudio
|-- coco(dataset)
|   |-- annotations
|       |-- instances_train2017.json
|       |-- instances_val2017.json
|   |-- train2017
|   |-- val2017
|   |-- test2017
|-- EfficientDet(repo)
```

## 训练

### 单卡训练


```python
python train.py -c 0 -p coco --batch_size 8 --lr 1e-5
```

`-c X`表明`efficientdet-dX`
![](https://ai-studio-static-online.cdn.bcebos.com/f41ba6a639be4afca746c731178b61a9e03719f9fb2f46c49544038e185c1acd)



### 验证

确保已安装`pycocotools`和`webcolors`
```
pip install pycocotools webcolors
```


```python
python coco_eval.py -p coco -c 0
```

你需要将权重文件下载至`weights`文件夹下，或者使用`-w`手动指定权重路径

#### 验证结果如下所示
**所有完整验证结果可在`EfficientDet/benchmark/coco_eval_result.txt`下查看**

| coefficient | pth_download | GPU Mem(MB) |mAP 0.5:0.95(this repo) | mAP 0.5:0.95(official) |
| :-----: | :-----: | :------: | :------: | :------: |
| D0 | [efficientdet-d0.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d0.pdparams) | 1049 |33.1 | 33.8
| D1 | [efficientdet-d1.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d1.pdparams) | 1159 |38.8 | 39.6
| D2 | [efficientdet-d2.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d2.pdparams) | 1321 |42.1 | 43.0
| D3 | [efficientdet-d3.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d3.pdparams) | 1647 |45.6 | 45.8
| D4 | [efficientdet-d4.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d4.pdparams) | 1903 |48.5 | 49.4
| D5 | [efficientdet-d5.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d5.pdparams) | 2255 |50.0 | 50.7
| D6 | [efficientdet-d6.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d6.pdparams) | 2985 |50.7 | 51.7
| D7 | [efficientdet-d7.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d7.pdparams) | 3819 |52.6 | 53.7
| D7X | [efficientdet-d8.pdparams](https://github.com/GuoQuanhao/EfficientDet-Paddle/releases/download/pretrainedmodel/efficientdet-d8.pdparams) | 3819 |53.8 | 55.1

### 推理


```python
python efficientdet_test.py
```

注意到你需要手动更改中第`17`行`compound_coef = 8`来指定`efficientdet-dX`

**部分模型推理结果如下所示**

<center><font face="楷体" size=4>原始图像&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;官方</font><font face="Times New Roman" size=4>efficientdet-d0</font><font face="楷体" size=4>预测图像</font></center>

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/086552e84f5647888373676612f34b583e27a8a63386457f971e5cdb824d964b" width="400"/><img src="https://ai-studio-static-online.cdn.bcebos.com/701aca4bbfc8410b8c7d5d824ae93dbb885eaacb115347458f579f201ee18088" width="400"/></center>


<center><font face="楷体" size=4>本项目</font><font face="Times New Roman" size=4>efficientdet-d0</font><font face="Times New Roman" size=4>预测图像&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本项目</font><font face="Times New Roman" size=4>efficientdet-d8</font><font face="Times New Roman" size=4>预测图像</font></center>

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c063b3deaa1a42b1abee5dcb52c1ab8e1c74e557802341ce92c7f6528e098de4" width="400"/><img src="https://ai-studio-static-online.cdn.bcebos.com/c2d3a5a474cd4da9942af092c05c38f9af0a12551e8b4812981852d54154c460" width="400"/></center>


```python
python efficientdet_test_videos.py
```

**以`efficientdet-d0`为例，测试效果如下**

![results](https://user-images.githubusercontent.com/49911294/136463881-928ee08f-6a03-4966-9b22-7e224523c813.gif)


## LOGO识别*【已有的COCO格式数据集】

整个日志已挂载在**benchmark**文件夹下
**采用的数据集已经具备COCO格式**，数据集已上传至[AIStudio](https://aistudio.baidu.com/aistudio/datasetdetail/113432)
解压缩后工程目录大致如下，可根据实际情况修改
```
home/aistudio
|-- logo(dataset)
|   |-- annotations
|       |-- instances_train.json
|       |-- instances_val.json
|   |-- train
|   |-- val
|-- EfficientDet(repo)
```
### 训练
利用`-p`指定工程，`-c`指定模型，`--load_weights`加载`COCO`预训练模型，由于数据集简单，设置`--head_only True`其余参数可自行决定


```python
python train.py -c 0 -p logo --head_only True --lr 5e-3 --batch_size 32 --load_weights weights/efficientdet-d0.pdparams  --num_epochs 10 --save_interval 100
```

### 评估


```python
python coco_eval.py -c 0 -p logo -w "logs/logo/efficientdet-d0_29_1100.pdparams"
```

```
DONE (t=0.08s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.761
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.655
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.589
```


### 预测


```python
python efficientdet_test.py
```

**注意到：你需要修改自己的图片路径和标签列表**

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c2da7f1855f04bc58f9c18e95a2de7176d077289b13a469f9391b6080addda33" width="400"/></center>


## 自定义数据集训练*【从零自定义数据集】

**注意到，此自定义数据集训练采用AIStudio Tesla 32G，本地训练你需要调整部分代码文件路径，采用的标注格式为VOCxml2COCOjson**

### 数据集介绍
本次使用的数据集来自于Kaggle平台开放数据集扑克牌识别，数据集已上传至[AIStudio](https://aistudio.baidu.com/aistudio/datasetdetail/113199)
**数据集文件具有如下目录，其标注格式为`VOC xml`**
```
scenes
|-- generated
|   |-- 000090528.jpg
|   |-- ...
|   |-- 000233645.jpg
|-- xml
|   |-- 000090528.xml
|   |-- ...
|   |-- 000233645.xml
```
利用labelImg调值PascalVOC可查看标注情况如下所示，其主要标注扑克牌正对角线的标签，labelImg我已打包成exe文件可在[此处](https://download.csdn.net/download/qq_39567427/19911797?spm=1001.2014.3001.5501)下载，或参考[此链接](https://blog.csdn.net/qq_39567427/article/details/104538053?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163497805516780357257035%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=163497805516780357257035&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v2~rank_v29-2-104538053.pc_v2_rank_blog_default&utm_term=%E6%89%93%E5%8C%85&spm=1018.2226.3001.4450)自行打包

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/880629d073bf442990e65f0d552e717e994e68681b544eb293442166a608a702" width="400"/></center>

### 数据集处理

为了能够训练吗，我提供了`puke_dataset.sh`脚本，进入`EfficientDet`文件夹运行即可自动处理数据
```shell
cd /home/aistudio/EfficientDet
bash puke_dataset.sh
```

![](https://ai-studio-static-online.cdn.bcebos.com/97fe6411e27944278104b5b95f22840e93527d48967d486aa2fa0de9fe1fa1aa)

这里简要介绍`puke_dataset.sh`执行机制，`puke_dataset.sh`执行顺序如下：
 - 解压数据集`puke.zip`
 - 运行`dataset_prepare.py`，划分训练和验证集(你可以自行调整代码划分出测试集)
 - 运行`xml2coco.py`转换标签格式
 - 删除原始解压数据集，减少硬盘占用

### 训练
利用`-p`指定工程，`-c`指定模型，`--load_weights`加载`COCO`预训练模型，其余参数可自行决定


```python
python train.py -c 0 -p puke --batch_size 32 --lr 1e-3 --num_epochs 10 --load_weights ./weights/efficientdet-d0.pdparams
```

### 预测


```python
python efficientdet_test.py
```

**注意到：你需要修改自己的图片路径和标签列表**

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/452d4eece5e14e34b6aea814e53a30f2c30cc362d30f433f91bc1cecdd48525b" width="400"/></center>


### TODO

- **多卡训练(Coming soon)**
- **YOLO格式转COCO训练(Coming soon)**
- **EfficientNet Pretrained Weights(Coming soon)**

#### [GitHub地址](https://github.com/GuoQuanhao/EfficientDet-Paddle)

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- |
| 学校        | 电子科技大学研2020级     |
| 研究方向     | 计算机视觉             |
| BLOG主页        | [DeepHao的 blog 主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [DeepHao的 github 主页](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
