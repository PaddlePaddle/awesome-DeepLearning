# 人脸表情迁移

## 1. 项目概述

深度学习中很多技术都可以用来做一些极具趣味性的应用，比如之前全网火爆的 "蚂蚁呀嘿" 就可以使用PaddleGAN的动作迁移模型来实现。除此之外，如果给照片中的人物赋予一个规定动作，那么该模型还可以让老照片中的故人“动起来”，做出你指定的动作，帮助人们以全新的视角看看过去的照片。

本案例基于飞桨PaddleGAN实现人脸表情迁移，提供预训练模型，无需训练可直接使用，快速实现表情迁移效果。

首先用它来 “复活” 一下祖师爷——现代计算机科学的先驱人工智能之父阿兰·图灵：

<center><image src="https://ai-studio-static-online.cdn.bcebos.com/ee584fdc07f9464d9277ff3df27ff81e1f92058b66ec42d29471abfd97fcdb14" width=400></center>


同时，也支持多位故人同时 “动起来”，比如下面这张居里夫妇的照片：
<center><image src="https://ai-studio-static-online.cdn.bcebos.com/373e77ce8038489fb71dac09b74c06c602c8d34cef6c49799bef2eea9435b9a7" width=400></center>


本项目AI Studio链接：[人脸表情迁移](https://aistudio.baidu.com/aistudio/projectdetail/2467323)

**如果您觉得本案例对您有帮助，欢迎Star收藏一下，不易走丢哦~，链接指路：** [awesome-DeepLearning](
https://github.com/PaddlePaddle/awesome-DeepLearning)



## 2. 解决方案

本项目采用PaddleGAN动作迁移模型中的人脸表情迁移模型First Order Motion来实现图像动画（Image Animation）任务，即输入一张源图片和一个驱动视频，源图片中的人物会做出驱动视频中的动作。

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/835af92a6dd54304960fbef7818c248aaf3ec4e6d9a64c2ab0ac0cb758fcef22" width="400"/>
</div><br></br>

图片来源：https://aliaksandrsiarohin.github.io/first-order-model-website/


如上图所示，源图像（第一列图片）中包含一个主体，驱动视频（第一行图片）包含一系列动作。给定源图片和驱动视频后，通过First Order Motion模型会生成一个新的视频，其主体是源人物，新生成的视频中源人物的表情由驱动视频中的表情决定。

下图简单的展现了其中的原理：
<div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/127443878-b9369c1a-909c-4af6-8c84-a62821262910.gif" width="500"/>
</div><br></br>
只需准备一张图片和一段驱动视频，即可实现人脸表情迁移。



## 3. 数据准备

本项目提供了上图中使阿兰·图灵动起来的驱动视频和原始图片。数据保存在 `data` 文件夹下。驱动视频路径为 `data/driving_video.MOV`，原始图片路径为 `data/image.jpeg`。你也可使用自己准备的视频和照片。



## 4. 模型推理

### 下载PaddleGAN并安装相关包
在进行动作表情迁移前，需要先下载PaddleGAN并安装相关包。具体命令如下：


```python
# 从github/gitee上下载PaddleGAN的代码，只需在第一次运行项目时下载即可
# github下载可能速度较慢，推荐通过gitee下载
# git clone https://github.com/PaddlePaddle/PaddleGAN
git clone https://gitee.com/paddlepaddle/PaddleGAN.git
```

```python
# 这里使用PaddleeGAN develop版本
cd PaddleGAN
git checkout develop
```

```python
# 安装所需安装包
pip install -r requirements.txt
pip install imageio-ffmpeg
```

### 表情动作迁移

运行如下命令，实现表情动作迁移。其中，各参数的具体使用说明如下： 
- driving_video: 驱动视频，视频中人物的表情动作作为待迁移的对象。本项目中驱动视频路径为 "data/driving_video.MOV"，大家可以上传自己准备的视频，更换 `driving_video` 参数对应的路径;
- source_image: 原始图片，视频中人物的表情动作将迁移到该原始图片中的人物上。这里原始图片路径使用 "data/image.jpeg"，大家可以使用自己准备的图片，更换 `source_image` 参数对应的路径;
- relative: 指示程序中使用视频和图片中人物关键点的相对坐标还是绝对坐标，建议使用相对坐标，若使用绝对坐标，会导致迁移后人物扭曲变形;
- adapt_scale: 根据关键点凸包自适应运动尺度;
- ratio: 针对多人脸，将框出来的人脸贴回原图时的区域占宽高的比例，默认为0.4，范围为【0.4，0.5】

命令运行成功后会在ouput文件夹生成名为result.mp4的视频文件，该文件即为动作迁移后的视频。


```bash
python -u applications/tools/first-order-demo.py --driving_video ../data/driving_video.MOV  --source_image ../data/image.jpeg --ratio 0.4 --relative --adapt_scale --output  ../data
```

### 视频配乐
可以执行如下命令为视频配乐，生成的新视频保存为 `data/output.mp4`。


```bash
ffmpeg -y -i ../data/music.mov -i ../data/result.mp4 -strict -2 -q:v 1 ../data/output.mp4
```



## 资源

更多资源请参考：

* 更多深度学习知识、产业案例，请参考：[awesome-DeepLearning](https://github.com/paddlepaddle/awesome-DeepLearning)

* 更多生成对抗网络实现，请参考：[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)

* 更多学习资料请参阅[飞桨深度学习平台](https://www.paddlepaddle.org.cn/?fr=paddleEdu_aistudio)
