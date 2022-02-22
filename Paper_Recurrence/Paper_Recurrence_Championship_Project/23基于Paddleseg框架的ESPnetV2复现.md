# 论文简介

作者提出了一个轻量级、效率高的通用卷积神经网络模型ESPNetV2，为了减少参数，ESPNetV2使用了分组卷积(Group Convolution)和深度可分离空洞卷积(Depth-wise dilated separable convolution)，作者利用分组卷积和深度可分离空洞卷积设计了非常精巧的EESP模块，能够大大减少模型的参数并且保持模型的性能。
## EESP模块
![EESP](https://ai-studio-static-online.cdn.bcebos.com/f6d513b9519a4d2fa38b0afcf5dd710cedd9985f36d54f458c29317ab644aebf)  
在上图中，Conv-n表示n * n的标准卷积，GConv-n表示n * n的分组卷积，DConv-n表示n * n的空洞卷积，DDConv-n表示n * n的深度可分离空洞卷积  
从上图可以看出EESP相对ESP的改进有以下几点，极大的减少了参数数量：  
1、使用分组卷积（GConv-1）替代了标准卷积（DConv-1）  
2、使用深度可分离空洞卷积（DDConv-3）替代了空洞卷积（DConv-3）  
3、EESP在concat后添加了一个分组卷积  
ESP和EESP的参数数量比为：  
$$\ \frac{Md + n^2d^2K}{\frac{Md}{g} + (n^2+d)dK}$$  
当M=240，g=K=4,d=M/K=60时，EESP参数数量时ESP参数数量的1/7，极大减少了参数数量。
## Strided EESP模块
![Strided EESP](https://ai-studio-static-online.cdn.bcebos.com/2bc2aabc988a4f21a857a5d6bcae72d217932bb509b848779d2d1bc6c6991224)  
为了在多尺度下有效的学习特征，对EESP模块进行了修改，得到了上图的Strided EESP模块，修改的内容如下：  
1、深度可分离空洞卷积步长修改为2  
2、Identity connection中添加步长为2的平均池化  
3、元素相加改为concat，来扩充特征维度  
4、从输入的图片引入shortcut，使用平均池化减少特征图大小  

## 网络结构
![网络结构](https://ai-studio-static-online.cdn.bcebos.com/64a15fd094e3432a85fa42bb6c553dda2c08489142144652b672753edb3d7f1d)  
ESPNetV2的网络结构如上图所示。
## 实验
![Experiment](https://ai-studio-static-online.cdn.bcebos.com/fa337dcc35bf46d2b1ef883541dd6b6ae303693984c44eff96bb30db97dc287a)  
作者在四种任务上测试了ESPNetV2的性能，分别是目标分类、语义分割、目标检测、语言模型。本项目仅复现ESPNetV2在语义分割的结果，故仅展示其在语义分割的实验结果。本次复现使用模型的最后一列作为backbone。

# 项目介绍
本项目为第四届百度论文复现赛ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network复现结果，本项目基于PaddleSeg实现，并复现了论文实验结果，非常感谢百度提供的比赛平台以及GPU资源。

## 环境介绍
硬件： Tesla V100 * 4  
框架： paddlepaddle==develop (paddlepaddle version 2.1.2版本的CrossEntropyLoss函数，当参数包含ignore_index 和 weight时有bug，使用开发版本)  

## 参数调试
1、优化器：论文中使用SGD优化器，但是论文提供的代码使用的Adam优化器，这里使用Adam优化器；  
2、学习率：使用PolynomialDecay学习率衰减策略，测试了学习率0.01和0.001的效果，当学习率为0.01时，损失收敛很慢，故采用初始学习率为0.001；  
3、训练时长：Tesla V100 * 4， batch_size为8，120k次迭代，训练总时长为21.5h。  

## 复现经验
1、论文中的实验说明和提供的源码可能有不同之处，需要仔细阅读源码，必要时与作者联系原因；  
2、论文提供的代码除了网络模型结构部分，其他部分也要仔细阅读，尤其是数据预处理的方法以及训练模型的方法；  
3、论文提供的源码使用的框架是pytorch，需要注意不同框架模型转换的细节（尤其是部分函数的不同）。  

## 复现结果
| |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ESPNetV2|120k|adam|1024x512|8|CityScapes|32G|4|0.6956|[espnet_cityscapes_1024_512_120k_x2.yml](configs/espnet_cityscapes_1024_512_120k_x2.yml)|

## EESP代码简介
```
class EESP(nn.Layer):
    """
    EESP block, principle: reduce -> split -> transform -> merge

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2.
        branches (int): Number of branches.
        kernel_size_maximum (int): A maximum value of receptive field allowed for EESP block.
        down_method (str): Down sample or not, only support 'avg' and 'esp'. (equivalent to stride is 2 or not)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 branches=4,
                 kernel_size_maximum=7,
                 down_method='esp'):
        super(EESP, self).__init__()
        assert out_channels % branches == 0, "out_channels should be a multiple of branches."
        assert down_method in ['avg', 'esp'], "down_method only support 'avg' or 'esp'."
        self.in_channels = in_channels
        self.stride = stride

        in_branch_channels = int(out_channels / branches)
        self.group_conv_in = ConvBNPReLU(in_channels, in_branch_channels, 1, stride=1, groups=branches)

        # 卷积分支，使用不同的dilation
        map_ksize_dilation = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.kernel_sizes = []
        for i in range(branches):
            kernel_size = 3 + 2 * i
            kernel_size = kernel_size if kernel_size <= kernel_size_maximum else 3
            self.kernel_sizes.append(kernel_size)
        self.kernel_sizes.sort()

        self.spp_modules = nn.LayerList()
        for i in range(branches):
            dilation = map_ksize_dilation[self.kernel_sizes[i]]
            self.spp_modules.append(
                nn.Conv2D(in_branch_channels,
                          in_branch_channels,
                          kernel_size=3,
                          padding='same',
                          stride=stride,
                          dilation=dilation,
                          groups=in_branch_channels,
                          bias_attr=False)
            )
        self.group_conv_out = ConvBN(out_channels, out_channels, kernel_size=1, stride=1, groups=branches)
        self.bn_act = BNPReLU(out_channels)
        self._act = nn.PReLU()
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):
        group_out = self.group_conv_in(x)  # reduce feature map dimensions
        output = [self.spp_modules[0](group_out)]

        # compute the output for each branch and hierarchically fuse them
        for k in range(1, len(self.spp_modules)):
            output_k = self.spp_modules[k](group_out)
            output_k = output_k + output[k - 1]     # HFF
            output.append(output_k)

        group_merge = self.group_conv_out(self.bn_act(paddle.concat(output, axis=1)))   # merge
        del output

        # if down-sampling, then return the merged feature map.
        if self.stride == 2 and self.downAvg:
            return group_merge

        # residual link
        if group_merge.shape == x.shape:
            group_merge = group_merge + x
        return self._act(group_merge)
```

## 项目实践
按照下述流程，训练并验证ESPNetV2。


```python
#step 1: git clone, 本项目已经clone，可以跳过此步骤
%cd /home/aistudio/
!git clone https://hub.fastgit.org/justld/EspnetV2_paddle.git
```


```python
#  step 2: 解压数据集
%cd /home/aistudio/data/data64550
!tar -xf cityscapes.tar
%cd /home/aistudio
```


```python
# step 3: 安装develop版本的paddlepaddle-gpu，如果使用cpu请忽略此步骤
# 说明：由于paddlepaddle==2.1.2版本的交叉熵损失函数输入权重时存在bug，故需要更换develop版本
!python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```


```python
# step 4: 计算交叉熵损失的权重，将其放入配置文件中，此计算过程比较慢，可以直接使用configs目录下的相关参数，如果使用提供的配置文件，请忽略此步骤
%cd /home/aistudio/EspnetV2_paddle
!python compute_classweight.py
```


```python
# step 5: 训练， 需要修改配置文件中的数据路径, 示例：dataset_root: /home/aistudio/data/data64550/cityscapes
%cd /home/aistudio/EspnetV2_paddle
!python train.py --config /home/aistudio/EspnetV2_paddle/configs/espnet_cityscapes_1024_512_120k_x2.yml --do_eval --use_vdl --log_iter 10 --save_interval 2000 --save_dir output
```


```python
# step 6: 验证，由于ESPNetV2模型比较小，训练好的模型参数已经放入github output目录下，直接使用即可
%cd /home/aistudio/EspnetV2_paddle
!python val.py --config /home/aistudio/EspnetV2_paddle/configs/espnet_cityscapes_1024_512_120k_x2.yml --model_path /home/aistudio/EspnetV2_paddle/output/scale_x2/best_model/model.pdparams
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
