# 精准唇形合成

## 1. 项目概述
你可曾想过让苏轼念诗，让蒙娜丽莎播新闻，让新闻主播唱Rap...? 最近，网络上爆火的苏轼念诗视频就是利用深度学习技术使宋代诗人苏轼活过来，穿越千年，亲自朗诵其著名古诗。

本案例基于飞桨PaddleGAN实现唇形合成，提供预训练模型，无需训练可直接使用，快速实现唇形合成效果。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/16d0b24fdc5c451395b3b308cf27b59bd4b024366b41457dbb80d0105f938849" width=500></center>



本项目AI Studio链接：[精准唇形合成](https://aistudio.baidu.com/aistudio/projectdetail/2504802)

**如果您觉得本案例对您有帮助，欢迎Star收藏一下，不易走丢哦~，链接指路：** [awesome-DeepLearning](
https://github.com/PaddlePaddle/awesome-DeepLearning)



## 2. 解决方案

唇形合成模型 Wav2lip 实现了任务口型与输入语音同步，俗称「对口型」。不仅可以让静态图像「说话」，还可以直接将动态视频进行唇形转换，输出与目标语音相匹配的视频，实现自制视频配音。

Wav2lip 实现唇形与语音精准同步的关键在于，它采用了唇形判别器来强制生成器产生准确而逼真的唇部运动。此外，考虑到时间相关性，Wav2Lip在判别器中使用了多个连续帧，并通过视觉质量损失来提升视觉质量。Wav2Lip适用于任意人脸、任意语言，并对任意视频都可达到很好的效果，可以无缝与原始视频融合。

如想实现唇形合成功能，只需准备一张图片或一段视频以及一段音频，音频用于驱动唇形合成，而图片/视频中的人物则根据此音频进行唇形合成。通俗来说，图片/视频文件提供想说话的人，音频文件提供想让这个人说什么。



## 3.数据准备
本项目提供了蒙娜丽莎的图片和一段新闻播报音频。数据保存在 data 文件夹下。图片路径为 data/picture.jpeg，音频文件路径为 data/audio.m4a。你也可以准备自己需要的图片/视频以及音频文件。



## 4. 模型推理

### 下载PaddleGAN并安装相关包


```python
# 从github/gitee上下载PaddleGAN的代码，只需在第一次运行项目时下载即可
# github下载可能速度较慢，推荐通过gitee下载
#git clone https://github.com/PaddlePaddle/PaddleGAN
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
```

### 唇形动作合成
运行如下命令，实现唇形合成。其中，参数使用说明如下：

- face: 图片/视频，其中的人物唇形将根据音频进行唇形合成。
- audio: 驱动唇形合成的音频。
- outfile: 指定生成的视频文件的保存路径及文件名

本项目支持大家上传自己准备的视频和音频，合成任意想要的配音视频。程序运行完成后，会生成outfile参数指定的视频文件。


```python
python applications/tools/wav2lip.py --face ../data/picture.jpeg --audio ../data/audio.m4a --outfile ../data/output.mp4
```



## 资源

更多资源请参考：

* 更多深度学习知识、产业案例，请参考：[awesome-DeepLearning](https://github.com/paddlepaddle/awesome-DeepLearning)

* 更多生成对抗网络实现，请参考：[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)

* 更多学习资料请参阅[飞桨深度学习平台](https://www.paddlepaddle.org.cn/?fr=paddleEdu_aistudio)
