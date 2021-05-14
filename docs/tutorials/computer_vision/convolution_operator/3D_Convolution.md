# 3D卷积（3D Convolution）

## 一、3D卷积

标准卷积也就是2D卷积，计算方式如 **图1** 所示。在2D卷积中，卷积核在图片上延宽和高两个维度滑动，每次滑动图像中对应位置内的元素与卷积核中的参数进行乘加计算，得到输出特征图中的一个值。

<center><img src="https://raw.githubusercontent.com/lvjian0706/Deep-Learning-Img/master/CNN/Convolution/3D_Convolution/img/2D_Convolution.png" width = "700"></center>
<center><br>图1：2D卷积示意图</br></center>

2D卷积仅仅只考虑2D图片的空间信息，所以只适用于单张2D图片的视觉理解任务。在处理3D图像或视频时，网络的输入多了一个维度，输入由$(c,height,width)$ 变为了$(c,depth,height,width)$ ，其中$c$ 是通道数。因此，就需要卷积也作出相应的变换，由2D卷积变为3D卷积。

在2D卷积的基础上，3D卷积被提出。3D卷积在结构上较2D卷积多了一个维度，2D卷积的尺寸可以表示为$k_h\times{k_w}$ ，而3D卷积的尺寸可以表示为$k_h\times{k_w}\times{k_d}$ 。3D卷积的具体的计算方式与2D卷积类似，即每次滑动时与$c$ 个通道，大小为$(depth,height,width)$ 的图像做乘加运算，得到输出特征图中的一个值。

<center><img src="https://raw.githubusercontent.com/lvjian0706/Deep-Learning-Img/master/CNN/Convolution/3D_Convolution/img/3D_Convolution.png" width = "700"></center>
<center><br>图2：3D卷积示意图</br></center>



## 三、应用示例

3D卷积的主要应用就是视频理解和医疗图像领域。

在视频理解任务中，$k_d$ 就代表了时间维度，也就是每个3D卷积核处理的连续帧数。在视频理解领域的3D卷积计算中，首先会将$k_d$ 个连续帧组成一个3D的图像序列，然后在图像序列中进行卷积计算。3D卷积核会在$k_d$ 个连续帧上进行滑动，每次滑动$k_d$ 个连续帧中对应位置内的元素都要与卷积核中的参数进行乘加计算，最后得到输出特征图中的一个值。

3D CNN中^[1]^，使用了3D卷积对人体行为进行识别，网络结构如 **图3** 所示。网络只有3个卷积层、1个全连接层以及2个池化层。其中，前两个卷积层为3D卷积层，卷积核大小为$7\times{7}\times{3}$ 和$7\times{6}\times{3}$ ，也就是说每个卷积核处理3个连续帧中$7\times{7}$ 或$7\times{6}$ 大小的区域。

<center><img src="https://raw.githubusercontent.com/lvjian0706/Deep-Learning-Img/master/CNN/Convolution/3D_Convolution/img/3DCNN.png" width = "700"></center>
<center><br>图3：3D CNN网络结构</br></center>

由于该模型使用了3D卷积，使得其可以从空间和时间的维度提取特征，从而捕捉从多个连续帧中得到的运动信息。

## 参考文献

[1] [3D Convolutional Neural Networks for Human Action Recognition](http://users.eecs.northwestern.edu/~mya671/mypapers/ICML10_Ji_Xu_Yang_Yu.pdf)

