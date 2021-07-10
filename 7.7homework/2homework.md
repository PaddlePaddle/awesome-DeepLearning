## **损失函数方法补充**

**交叉熵损失函数**
交叉熵损失函数的标准形式如下：
$$
C=-\frac{1}{n}\sum_{x}{[y\ln a+(1-y)\ln(1-a)]}
$$
公式中x表示样本，y表示实际的标签，a表示预测的输出，n表示样本总数量。

特点：
（1）本质上是一种对数似然函数，可用于二分类和多分类任务中。

二分类问题中的loss函数（输入数据是softmax或者Sigmoid函数的输出）： 
$$
loss=-\frac{1}{n}\sum_{x}{[y\ln a+(1-y)\ln (1-a)]}
$$
多分类问题中的loss函数（输入数据是softmax或者Sigmoid函数的输出）： 
$$
loss=-\frac{1}{n}\sum_{i}{y_i\ln a_i}
$$

 （2）当使用Sigmoid作为激活函数的时候，常用交叉熵损失函数而不用均方误差损失函数，因为它可以完美解决平方损失函数权重更新过慢的问题，具有“误差大的时候，权重更新快；误差小的时候，权重更新慢”的良好品质。

```python
import numpy as np

def cross_entropy(Y, P):
    """Cross-Entropy loss function.
    以向量化的方式实现交叉熵函数
    Y and P are lists of labels and estimations
    returns the float corresponding to their cross-entropy.
    """
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) / len(Y)

```

**0-1损失函数**
0-1损失是指预测值和目标值不相等为1，否则为0：

特点：
（1）0-1损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用

（2）感知机就是使用的这种损失函数，但是因为相等的条件过于严格，可以放宽条件，即满足预测值和目标值的差值小于一定值的时候，就认为是相等的。

```python
def 0-1_loss(Y,P):
	if abs(P-Y)<T:
		return 0
	else:
		return 1
```

**Hinge损失函数**
Hinge损失函数标准形式如下： 
$$
L(y,f(x))=max(0,1-yf(x))
$$


特点：
（1）hinge损失函数表示如果分类正确，损失为0，否则损失就为1-yf(x)。SVM就是使用这个损失函数。

（2）一般的f(x)是预测值，在-1到1之间，y是目标值（-1或1）。其含义是，f(x)的值在-1和+1之间就可以了，并不鼓励|f(x)|>1，即并不鼓励分类器过度自信，让某个正确分类的样本距离分割线超过1并不会有任何奖励，从而使分类器可以更专注于整体的误差。

（3）健壮性相对较高，对异常点、噪声不敏感，但它没有太好的概率解释。
```python
from sklearn import svm
from sklearn.metrics import hinge_loss
X = [[0], [1]]
y = [-1, 1]
model = svm.LinearSVC(random_state=0)
model.fit(X, y)

pred_decision = model.decision_function([[-2], [3], [0.5]])
print(pred_decision)
print(hinge_loss([-1, 1, 1], pred_decision))

print()

X = [[0,0], [1,1], [2,2], [3,3]]
Y = [-1, -1, -1, 1]
model = svm.LinearSVC()
model.fit(X, Y)
pred_decision = model.decision_function([[-1, -1], [4, 4]])
print(pred_decision)
y_true = [0, 3]
print(hinge_loss(y_true, pred_decision))
log对数损失函数
log对数损失函数的标准形式如下： $$ L(Y,P(Y|X))=-\log P(Y|X) $$

```
特点：
（1）log对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，如果需要知道结果属于每个类别的置信度，那它非常适合。

（2）健壮性不强，相比于hinge loss对噪声更敏感。

（3）逻辑回归的损失函数就是log对数损失函数。
```python
def logloss(y_true, y_pred, eps=1e-15):
    import numpy as np

    # Prepare numpy array data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert (len(y_true) and len(y_true) == len(y_pred))
    
    # Clip y_pred between eps and 1-eps
    p = np.clip(y_pred, eps, 1-eps)
    loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1-p))
    
    return loss / len(y_true)
```
## 池化方法补充

**Overlapping Pooling**(重叠池化)
alexnet中提出和使用。 相对于传统的no-overlapping pooling，采用Overlapping Pooling不仅可以提升预测精度，同时一定程度上可以减缓过拟合。 相比于正常池化（步长s=2，窗口z=2） 重叠池化(步长s=2，窗口z=3) 可以减少top-1, top-5分别为0.4% 和0.3%；重叠池化可以避免过拟合。

```python
Mat pooling(Mat img, int grid, int overlap)
{
	Mat pool_img = Mat((int)((img.rows - 1) / overlap) + 1, (int)((img.cols - 1) / overlap)+1, CV_8UC1);
	for (int col = 0,pool_col=0; col < img.cols; col+= overlap)
	{
		for (int row = 0,pool_row=0; row < img.rows; row+= overlap)
		{
			int minCol = min(col + overlap, img.cols);
			int maxData = 0;
			for (int poolX = col; poolX < minCol; poolX++)
			{
				int minRow = min(row + overlap, img.rows);
				for (int poolY = row; poolY<minRow; poolY++)
				{
					if (img.at<uchar>(poolY, poolX)>maxData)
					{
						maxData = img.at<uchar>(poolY, poolX);
					}
				}
			}
			pool_img.at<uchar>(pool_row, pool_col) = maxData;
			pool_row++;
		}
		pool_col++;
	}
	return pool_img;
}
```
**Max pooling**（最大池化）和 **mean-pooling** （平均池化）
mean-pooling，即对邻域内特征点只求平均，max-pooling，即对邻域内特征点取最大。根据相关理论，特征提取的误差主要来自两个方面：（1）邻域大小受限造成的估计值方差增大；（2）卷积层参数误差造成估计均值的偏移。一般来说，mean-pooling能减小第一种误差，更多的保留图像的背景信息，max-pooling能减小第二种误差，更多的保留纹理信息。Stochastic-pooling则介于两者之间，通过对像素点按照数值大小赋予概率，再按照概率进行亚采样，在平均意义上，与mean-pooling近似，在局部意义上，则服从max-pooling的准则。

**最大池化**

```python
import numpy as np
def max_pooling(det,pool_param):
    M,N,H,W=det.shape()
    H1,W1,stride=pool_param['pool_heaight'],pool_param['pool_width',pool_param['stride']]
    H2=int(（H-H1）/stride)+1
    W2=int(（W-W1）/stride)+1
    out=np.zeros(M,N,H,W)
    for i in range(len(H2)):
        for j in range(len(W2)):
            out[...,i,j]=np.max(det[...,i*stride:i*stride+H2,j*stride:j*stride+W2],axis=(2,3))
    result=(det,out,pool_param)
    return out,result
np.max()
```
**平均池化**

```python
#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H
#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;

image get_avgpool_image(avgpool_layer l);

// 构造平均池化层函数   
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);

void resize_avgpool_layer(avgpool_layer *l, int w, int h);

// 平均池化层前向传播函数
void forward_avgpool_layer(const avgpool_layer l, network net);

// 平均池化层后向传播函数
void backward_avgpool_layer(const avgpool_layer l, network net);

#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network net);
void backward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif
```
**空间金字塔池化**
使得任意大小的特征图都能够转换成固定大小的特征向量，这就是空间金字塔池化的意义（多尺度特征提取出固定大小的特征向量），送入全连接层。整体框架大致为：输入图像，卷积层提取特征，空间金字塔池化提取固定大小特征，全连接层。

```python
import math
import torch
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
    
        self.num_levels = num_levels
        self.pool_type = pool_type
    
    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
    
            # 选择池化方式 
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
    
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
```
## 数据增强方法补充

**翻转**
数据翻转是一种常用的数据增强方法，这种方法不同于旋转 180 这种方法是做一种类似于镜面的翻折。

```python
import math
import torch
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
    
        self.num_levels = num_levels
        self.pool_type = pool_type
    
    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
    
            # 选择池化方式 
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
    
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
```
**缩放**
图像可以被放大或缩小。放大时，放大后的图像尺寸会大于原始尺寸。大多数图像处理架构会按照原始尺寸对放大后的图像 进行裁切。

```python
import cv2
from PIL import Image
import math
import numpy as np
import os
import pdb
import xml.etree.ElementTree as ET
from skimage import transform
from skimage.io import imread, imsave


class ImgAugemention():
    def __init__(self):
        self.formats = ['.png', '.jpg', '.jpeg']

    def zoom(self, img, p1x, p1y, p2x, p2y):
        w = img.shape[1]
        h = img.shape[0]    # print(w, h)    [1280, 720]
        crop_p1x = max(p1x, 0)
        crop_p1y = max(p1y, 0)
        crop_p2x = min(p2x, w)
        crop_p2y = min(p2y, h)
     
        cropped_img = img[crop_p1y:crop_p2y, crop_p1x:crop_p2x]
     
        x_pad_before = -min(0, p1x)
        x_pad_after = max(0, p2x - w)
        y_pad_before = -min(0, p1y)
        y_pad_after = max(0, p2y - h)
     
        padding = [(y_pad_before, y_pad_after), (x_pad_before, x_pad_after)]
        is_colour = len(img.shape) == 3
        if is_colour:
            padding.append((0, 0))  # colour images have an extra dimension
     
        padded_img = np.pad(cropped_img, padding, 'constant')
        return transform.resize(padded_img, (h, w))
     
    def vertical_xml(self, src, loc, zoom_loc):
        w = src.shape[1]
        h = src.shape[0]
        w1 = abs(zoom_loc[2] - zoom_loc[0])
        h1 = abs(zoom_loc[3] - zoom_loc[1])
        n_xmin = int(abs(loc[0] - zoom_loc[0]) * w / w1)
        n_ymin = int(abs(loc[1] - zoom_loc[1]) * h / h1)
        n_xmax = n_xmin + int(abs(loc[2] - loc[0]) * w / w1)
        n_ymax = n_ymin + int(abs(loc[3] - loc[1]) * h / h1)
     
        return n_xmin, n_ymin, n_xmax, n_ymax
     
    def process_img(self, imgs_path, xmls_path, img_save_path, xml_save_path, zoom_loc):
        for img_name in os.listdir(imgs_path):
            # split filename and suffix
            n, s = os.path.splitext(img_name)
            img_path = os.path.join(imgs_path, img_name)
            img = imread(img_path)
            zoom = self.zoom(img, zoom_loc[0], zoom_loc[1], zoom_loc[2], zoom_loc[3])
            # 写入图像
            imsave(img_save_path + n + "_" + "v_stretch.jpg", zoom)
            xml_url = img_name.split('.')[0] + ".xml"
            xml_path = os.path.join(xmls_path, xml_url)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.iter('object'):
                loc = []
                for bnd in obj.iter('bndbox'):
                    for xmin in bnd.iter('xmin'):
                        loc.append(int(xmin.text))
                    for ymin in bnd.iter('ymin'):
                        loc.append(int(ymin.text))
                    for xmax in bnd.iter('xmax'):
                        loc.append(int(xmax.text))
                    for ymax in bnd.iter('ymax'):
                        loc.append(int(ymax.text))
     
                    n_xmin, n_ymin, n_xmax, n_ymax = self.vertical_xml(img, loc, zoom_loc)   # change locs in xml file
                    # update locs in xml file
                    for xmin in bnd.iter('xmin'):
                        xmin.text = str(n_xmin)
                    for ymin in bnd.iter('ymin'):
                        ymin.text = str(n_ymin)
                    for xmax in bnd.iter('xmax'):
                        xmax.text = str(n_xmax)
                    for ymax in bnd.iter('ymax'):
                        ymax.text = str(n_ymax)
                    bnd.set('updated', 'yes')
            # write into new xml file
            tree.write(xml_save_path + n + "_" + "v_stretch.xml")

 

if __name__ == "__main__":
    img_aug = ImgAugemention()
    root_dir_path1 = "E:\\singel_zoom\\tmp\\"
    root_dir_path2 = "E:\\singel_zoom1\\tmp\\"
    zoom_loc = [0, 40, 1280, 660]         # [0],[1]为左上角坐标， [2][3]为右下角坐标， 以此两点作伸缩变换(原图尺寸为(1280， 720))
                                          # zoom: [276, 6, 994, 633]  h_stretch: [250, 0, 900, 720]  v_stretch: [0, 40, 1280, 660]

    path_list = [""]                      # path_list = ["", "1\\", "2\\", "3\\", "4\\", "5\\"]
    for n in path_list:
        path_image = root_dir_path1 + n
        path_xml = root_dir_path1 + n
        path_image_dst = root_dir_path2 + n
        path_xml_dst = root_dir_path2 + n
        img_aug.process_img(path_image, path_xml, path_image_dst, path_xml_dst, zoom_loc)
```
**锐化**
在图像增强过程中，通常利用各类图像平滑算法消除噪声，图像的常见噪声主要有加性噪声、乘性噪声和量化噪声等。一般来说，图像的能量主要集中在其低频部分，噪声所在的频段主要在高频段，同时图像边缘信息也主要集中在其高频部分。这将导致原始图像在平滑处理之后，图像边缘和图像轮廓模糊的情况出现。为了减少这类不利效果的影响，就需要利用图像锐化技术，使图像的边缘变得清晰。

图像锐化处理的目的是为了使图像的边缘、轮廓线以及图像的细节变得清晰，经过平滑的图像变得模糊的根本原因是因为图像受到了平均或积分运算，因此可以对其进行逆运算(如微分运算)就可以使图像变得清晰。微分运算是求信号的变化率，由傅立叶变换的微分性质可知，微分运算具有较强高频分量作用。从频率域来考虑，图像模糊的实质是因为其高频分量被衰减，因此可以用高通滤波器来使图像清晰。但要注意能够进行锐化处理的图像必须有较高的性噪比，否则锐化后图像性噪比反而更低，从而使得噪声增加的比信号还要多，因此一般是先去除或减轻噪声后再进行锐化处理。
```python
void sharpen(const Mat &image,Mat result)
{
    /// 判断是否需要分配图像数据。如果需要，就分配
    result.create(image.size(),image.type());
    int nchannels = image.channels(); /// 获得通道数
      接下来的程序就是处理所有的行
    for (int j = 1 ; j < image.rows-1; j++) {
        const uchar* previous = image.ptr<const uchar>(j-1);   /// 上一行
        const uchar* current = image.ptr<const uchar>(j);      /// 当前行
        const uchar* next = image.ptr<const uchar>(j+1);   ///下一行
        uchar* output = result.ptr<uchar>(j);  /// 输出行
      /**** 应用锐化算子操作部分 ****/
            在这里，将每个像素的三个通道，合并成一个 矩阵，所以这里有  (image.cols-1)*nchannels 这个操作
        ///  需要注意一点的是： 对于彩色图片，当三个通道的时候，一定是BGR对应颜色像素进行操作，所以这里是 current[i-nchannels]和current[i+nchannels]
        for (int i = nchannels; i < (image.cols-1)*nchannels; i++) {    /// 循环三个通道，进行相应的处理
        *output++ = saturate_cast<uchar>(
                5*current[i] - current[i-nchannels] - current[i+nchannels]-previous[i]-next[i]);       /// 该值的5倍 减去 上下左右四个数
        }
    }
    /// 接下来将未处理的像素都设为0
    result.row(0).setTo(Scalar(0));
    result.row(result.rows -1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols-1).setTo(Scalar(0));
    imshow("HAHAH1",result);
}
```
## 图像分类方法综述

**Nearest Neighbor分类器**
图像分类数据集：CIFAR-10。一个非常流行的图像分类数据集是CIFAR-10。这个数据集包含了60000张32X32的小图像。每张图像都有10种分类标签中的一种。这60000张图像被分为包含50000张图像的训练集和包含10000张图像的测试集。

假设现在我们有CIFAR-10的50000张图片（每种分类5000张）作为训练集，我们希望将余下的10000作为测试集并给他们打上标签。Nearest Neighbor算法将会拿着测试图片和训练集中每一张图片去比较，然后将它认为最相似的那个训练集图片的标签赋给这张测试图片。上面右边的图片就展示了这样的结果。请注意上面10个分类中，只有3个是准确的。比如第8行中，马头被分类为一个红色的跑车，原因在于红色跑车的黑色背景非常强烈，所以这匹马就被错误分类为跑车了。

在本例中，就是比较32x32x3的像素块。最简单的方法就是逐个像素比较，最后将差异值全部加起来。换句话说，就是将两张图片先转化为两个向量和，然后计算他们的L1距离。

以图片中的一个颜色通道为例来进行说明。两张图片使用L1距离来进行比较。逐个像素求差值，然后将所有差值加起来得到一个数值。如果两张图片一模一样，那么L1距离为0，但是如果两张图片很是不同，那L1值将会非常大。



首先，我们将CIFAR-10的数据加载到内存中，并分成4个数组：训练数据和标签，测试数据和标签。在下面的代码中，Xtr（大小是50000x32x32x3）存有训练集中所有的图像，Ytr是对应的长度为50000的1维数组，存有图像对应的分类标签（从0到9）：
```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') 
```
作为评价标准，我们常常使用准确率，它描述了我们预测正确的得分。请注意以后我们实现的所有分类器都需要有这个API：train(X, y)函数。该函数使用训练集的数据和标签来进行训练。从其内部来看，类应该实现一些关于标签和标签如何被预测的模型。这里还有个predict(X)函数，它的作用是预测输入的新数据的分类标签。现在还没介绍分类器的实现，下面就是使用L1距离的Nearest Neighbor分类器的实现套路：
```python
import numpy as npclass NearestNeighbor(object):

 def __init__(self):
 	 pass
 def train(self, X, y):
  self.Xtr = X
  self.ytr = y
    
 def predict(self, X):
 	num_test = X.shape[0]
  	Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
  	for i in xrange(num_test):
   		distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
   		min_index = np.argmin(distances) # get the index with smallest distance
   		Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
  	return Ypred
```
用这段代码跑CIFAR-10，会发现准确率能达到38.6%。这比随机猜测的10%要好，但是比人类识别的水平（据研究推测是94%）和卷积神经网络能达到的95%还是差多了。

换句话说，我们依旧是在计算像素间的差值，只是先求其平方，然后把这些平方全部加起来，最后对这个和开方。在Numpy中，我们只需要替换上面代码中的1行代码就行：
```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```
注意在这里使用了np.sqrt，但是在实际中可能不用。因为求平方根函数是一个单调函数，它对不同距离的绝对值求平方根虽然改变了数值大小，但依然保持了不同距离大小的顺序。所以用不用它，都能够对像素差异的大小进行正确比较。如果你在CIFAR-10上面跑这个模型，正确率是35.4%，比刚才低了一点。

**k-Nearest Neighbor分类器**

与其只找最相近的那1个图片的标签，我们找最相似的k个图片的标签，然后让他们针对测试图片进行投票，最后把票数最高的标签作为对测试图片的预测。所以当k=1的时候，k-Nearest Neighbor分类器就是Nearest Neighbor分类器。从直观感受上就可以看到，更高的k值可以让分类的效果更平滑，使得分类器对于异常值更有抵抗力。



上面示例展示了Nearest Neighbor分类器和5-Nearest Neighbor分类器的区别。例子使用了2维的点来表示，分成3类（红、蓝和绿）。不同颜色区域代表的是使用L2距离的分类器的决策边界。白色的区域是分类模糊的例子（即图像与两个以上的分类标签绑定）。需要注意的是，在NN分类器中，异常的数据点（比如：在蓝色区域中的绿点）制造出一个不正确预测的孤岛。5-NN分类器将这些不规则都平滑了，使得它针对测试数据的泛化（generalization）能力更好（例子中未展示）。注意，5-NN中也存在一些灰色区域，这些区域是因为近邻标签的最高票数相同导致的（比如：2个邻居是红色，2个邻居是蓝色，还有1个是绿色）。

**Fisher判别**

(Fisher)判别是一种常用的监督分类方法。它的准则是“组间最大分离”的原则，即要求组间（类间）距离最大而组内（类内）的离散性最小，也就是要求组间均值差异最大而组内离差平方和最小。费歇尔判别是利用一判别函数来进行最小距离分类的。当选用一次函数作为判别函数时为线性判别，本节只讨论Fisher线性判别。

Fisher线性判别是以多维正态分布为基础的，有些特征变量，如波段比值，不是正态分布的，需要进行变化或采用其它的分类器，这一点对于所有要求特征变量是正态分布的分类方法都是一样的。

下面给出Fisher线性判别的步骤：

1. 把来自2类的训练样本集划分为2个子集 ![[公式]](https://www.zhihu.com/equation?tex=X_0) 和 ![[公式]](https://www.zhihu.com/equation?tex=X_1) ；

2. 计算各类的均值向量 ![[公式]](https://www.zhihu.com/equation?tex=u_i) ， ![[公式]](https://www.zhihu.com/equation?tex=i%3D0%2C1) ； ![[公式]](https://www.zhihu.com/equation?tex=u_i%3D%5Cfrac%7B1%7D%7BN_i%7D%5Csum_%7Bx%5Cin+X_i%7Dx) ；

3. 计算各类的类内离散矩阵 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bwi%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=i%3D0%2C1) ； ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bwi%7D%3D%5Csum_%7Bx%5Cin+X_i%7D%28x-u_i%29%28x-u_i%29%5ET) ；

4. 计算类内总离散矩阵 ![[公式]](https://www.zhihu.com/equation?tex=S_w%3DS_%7Bw0%7D%2BS_%7Bw1%7D) ；

5. 计算矩阵 ![[公式]](https://www.zhihu.com/equation?tex=S_w) 的逆矩阵 ![[公式]](https://www.zhihu.com/equation?tex=S_w%5E%7B-1%7D) ；

6. 求出向量 ![[公式]](https://www.zhihu.com/equation?tex=w%5E%7B%2A%7D%3DS_w%5E%7B-1%7D%28u_1-u_0%29) （此处采用拉格朗日乘子法解出Fisher线性判别的最佳投影方向)；

7. 判别函数为 ![[公式]](https://www.zhihu.com/equation?tex=y%3Dw%5E%7B%2A%5ET%7Dx) ；

8. 判别函数的阈值 ![[公式]](https://www.zhihu.com/equation?tex=w_0) 可采用以下两种方法确定：第一种是 ![[公式]](https://www.zhihu.com/equation?tex=w_0%3D%5Cfrac%7Bw%5E%7B%2A%5ET%7Du_0%2Bw%5E%7B%2A%5ET%7Du_1%7D%7B2%7D) ；第二种是 ![[公式]](https://www.zhihu.com/equation?tex=w_0%3D%5Cfrac%7BN_0w%5E%7B%2A%5ET%7Du_0%2BN_1w%5E%7B%2A%5ET%7Du_1%7D%7BN_0%2BN_1%7D) ；

9. 分类规则：比较 ![[公式]](https://www.zhihu.com/equation?tex=y) 值与阈值 ![[公式]](https://www.zhihu.com/equation?tex=w_0) 的大小，得出其分类。