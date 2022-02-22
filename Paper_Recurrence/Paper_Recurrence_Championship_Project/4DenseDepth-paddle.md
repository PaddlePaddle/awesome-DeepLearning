# 模型名称  
# DenseDepth  
## 一、简介
DenseDepth出自论文High Quality Monocular Depth Estimation via Transfer Learning，是一个用于单目深度估计的网络。

[High Quality Monocular Depth Estimation via Transfer Learning](https://github.com/ialhashim/DenseDepth.git)

![avatar](https://ai.bdstatic.com/file/307349CFDB0D4861B61F3A870E762237)
DenseDepth主要是通过 Densenet169 Encoder提取图像特征 + 双线性插值上采样 Decoder还原图像深度，中心思想是利用在ImageNet上训练好的backbone
进行迁移训练，此外在训练中通过SSIM光度误差、图像梯度平滑误差和深度估值误差来约束训练方向。

![avatar](https://ai.bdstatic.com/file/11F45144B28B46C59CA03DD076448CD3)
DenseDepth在NYU depth数据集上的单目深度估计精度达到了当时的SOTA表现
## 二、复现精度
| | a1 | a2 | a3 | rel | rms | log10  
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|
原repo | 0.895 | 0.980 | 0.996 | 0.103 | 0.390 | 0.043 |
paddle复现| 0.896 | 0.981 | 0.995 | 0.106 | 0.456 | 0.044 |

## 三、数据集
论文采用的是nyu depth数据集进行训练，链接如下：
[nyu_depth](https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo?usp=sharing)

数据集大小为4.1G，包含50k张训练RGB图片和1308张测试RGB图片以及对应的深度图depth，此外还有nyu_train.csv和nyu_test.csv文件描述图片的位置
## 四、环境依赖
训练和测试时的环境依赖如下：  
```
paddlepaddle-gpu>=2.0
tensorboardX
```

训练时需要GPU RAM>=9GB(batch_size=4)   RAM>=18GB(batch_size=8)

测试可使用CPU  

## 五、快速开始
### 直接使用训练好的模型进行测试和评估
densenet预模型链接如下：  
链接：https://pan.baidu.com/s/1KUPnjUgpG40VSDLBHEcRIQ  
提取码：zid4  
将densenet模型放在model/中  

训练好的DenseDepth模型如下：  
链接：https://pan.baidu.com/s/1f1lYptz3xVMs3mJKvVgrUw  
提取码：as0z  
将DenseDepth模型放在logs/中  

然后就可以直接进行图片测试和精度评估：  
```
# 使用图片测试，首先将需要测试的图片放入images文件夹中，然后输入以下命令进行测试
python predict.py

# 精度评估时，首先需要将nyu数据集放入data文件夹中，然后输入以下命令进行评估
python eval.py

# 如果需要自定义模型和图片加载的路径，可以修改configs/main.cfg中对应的eval和predict参数
```
### 使用densenet进行迁移训练
densenet预模型链接如下：  
链接：https://pan.baidu.com/s/1KUPnjUgpG40VSDLBHEcRIQ  
提取码：zid4  
将densenet模型放在model/中  

将数据集nyu_data.zip放入data/中  

修改configs/main.cfg(推荐使用默认参数)，然后开始训练：  
```
python train.py
```

## 六、代码结构与详细说明
### 代码结构
├─configs  
├─data  
├─images  
├─logs  
├─model  
├─results  
├─utils  
│  eval.py  
│  predict.py  
│  README.md  
│  README_cn.md  
│  requirement.txt  
│  train.py  

### 参数说明
configs/main.cfg 参数配置文件，其中train,predict,eval分别在训练、测试和评估时使用  

```
[train] 训练参数
# 训练轮数
epochs = 20
# 学习率
learning_rate = 0.0001
# 批个数
batch_size = 8

[eval] 评估参数
# densedepth模型的位置
weights_path = logs/DenseDepth_val_best.pdparams

[predict]
# densedepth模型的位置
weights_path = logs/DenseDepth_val_best.pdparams
# 测试图片文件夹位置
imagedir_path = images/
# 测试结果的颜色控制
color_map = gray
```

### 文件说明
```
configs/main.cfg   # 参数设置
data/data.py   # 包含nyu_data.zip读取、转化为paddle dataset格式、数据集预处理方法
images/   # 测试图片文件夹
logs/   # 训练时模型的存放位置\
model/densenet   # densenet backbone结构定义文件
model/model.py   # densedepth模型结构定义文件
results/   # 测试结果图片存放位置
utils/losses.py   # 训练时使用到的的loss定义文件
utils/utils.py    # 训练、测试和评估时用到的一些辅助工具

eval.py  # 评估文件
predict.py   # 测试文件
train.py   # 训练文件
```

## 七、模型信息
| 信息 | 描述 |
:-----:|:-----:|
作者 | Rui He |
日期| 2021.08 |
框架| paddle 2.1.0 |
应用场景| 深度估计 |
硬件支持| CPU、GPU |
模型下载链接| [DenseDepth：提取码as0z](https://pan.baidu.com/s/1f1lYptz3xVMs3mJKvVgrUw) |
repo地址| [DenseDepth-paddle](https://github.com/stunback/DenseDepth-paddle.git) |

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
