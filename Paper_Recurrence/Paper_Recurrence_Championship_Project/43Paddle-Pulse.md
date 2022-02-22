# 复现  PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models

## 一、简介

论文名称：PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models  

本项目基于paddlepaddle框架复现PULSE，该论文整体思路是利用一个预先训练好的GAN，然后通过不断迭代找到一个最优的latent vector使得生成的HR图片经过下采样能够与输入的LR图片最接近。该方法主要用来处理大因子的图像超分辨率问题，可以将模糊的照片秒变清晰。

**论文:**  https://arxiv.org/pdf/2003.03808v3.pdf

**参考项目：**

- [https://github.com/adamian98/pulse](https://github.com/adamian98/pulse)

**项目github地址：**

- github：[https://github.com/Martion-z/Paddle-PULSE](https://aistudio.baidu.com/aistudio/projectdetail/2255411)

## 二、复现结果

### 2.1 视觉效果

celeba HQ数据集中随机选取了20张图片（16x16），比较其torch版本和paddle版本的输出结果（1024x1024），均进行100次迭代
​  

![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwqt3a1fj60ps0extah02.jpg)  
![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwr0o65tj60pd0et75z02.jpg)  
（视觉效果都挺好）

### 2.2 NIQE指标

从celeba HQ数据集中随机选取了20张预处理好的图片（16x16），作为torch版本和paddle版本的输入，比较其输出图片（1024*1024）的平均NIQE值(越小越好)。  
以下为结果截图:  
**torch版本：average_NIQE=2.174**  
![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwbx6ua5j60q00lltc302.jpg)  
**paddle版本：average_NIQE=2.132**  
![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcx4g6u1hj60q50nkwki02.jpg)



## 三、数据集

输入的图片需要放置于input文件夹中，与[原参考代码](https://github.com/adamian98/pulse)一致，输入的图片为16x16大小的celabaHQ人脸数据集，附上数据集链接： [百度云盘](https://pan.baidu.com/s/1wGbZ4UxPDpQj2gV_Zq37pQ)  密码: mo0s

**已挂载数据集**  


## 四、环境依赖

- 硬件：GPU、CPU

- 框架：

  - PaddlePaddle >= 2.0.0

    ​

## 五、快速开始

**在线运行notebook**  

终端执行`python3 run.py`即可运行代码，算法通过不断迭代寻找最佳输出图像，输出结果（1024x1024）存在output1024文件夹中。  

  ​

## 六、代码结构与详细说明

### 6.1 代码结构

```
./Paddle-Pulse
|-- images               # 图片文件夹
       |--input          #输入文件夹
       |--output1024        #输出文件夹
|-- models                # 模型实现文件夹
       |--cache         #模型权重文件夹
       |--loss            #模型损失函数类文件夹
       |--utils            #模型工具类API文件夹
       |--pulse.py        #pulse网络结构
       |--stylegan_paddle.py    #stylegan网络结构
|-- utils                # 工具类API文件夹
|-- run.py                    #主函数调用所有类
|-- README.md            # 用户手册  
```

stylegan.pdparams为在FFQH数据集上预训练好的styleGan的生成器的权重  
gaussion_fit为在FFQH数据集上预训练好的styleGan的非线性映射网络  
run.py 为运行主函数  
stylegan_paddle.py 文件为styleGan的网络结构  
pulse.py 为论文PULSE提取的算法，利用预先训练好的gan不断迭代寻找最优图片  
loss.py 损失函数类
SphericalOptimizer.py 文件里为优化器类  
bicubic.py  双三次下采样类  
drive.py  驱动下载类  
niqe.py NIQE评价类

### 6.2 参数说明

可以通过命令行调节以下相关参数，具体如下：

| 参数         | 默认值        | 说明           |
| ---------- | ---------- | ------------ |
| input_dir  | input      | 输入图片的路径      |
| output_dir | Output1024 | 输出图片的存放路径    |
| batch_size | 1          | 每批次大小        |
| seed       | 0          | 随机种子         |
| eps        | 2e-3       | 目标最小损失       |
| opt_name   | adam       | 优化器类别        |
| steps      | 100        | 寻找最优图片时的迭代次数 |


## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息   | 说明                                       |
| ---- | ---------------------------------------- |
| 发布者  | 皮蛋瘦肉周                                    |
| 时间   | 2021.08                                  |
| 框架版本 | Paddle 2.0.2                             |
| 应用场景 | 图像超分辨率                                   |
| 支持硬件 | GPU、CPU                                  |
| 下载链接 | [预训练模型](https://pan.baidu.com/s/1zRvbGmt7IOMoWSmQQz-ZHA) 提取码：f35u |
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/2255411) |
