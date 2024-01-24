```python
# #
# # View dataset directory.
# # This directory will be recovered automatically after resetting environment.
# # 处理数据集，首次执行后注释
# import os
# if not os.path.exists("./data/place2"):
#     os.mkdir("./data/place2")
# if not os.path.exists("./data/mask"):
#     os.mkdir("./data/mask")

# !unzip -qa -d /home/aistudio/data/mask/ /home/aistudio/data/data105125/center_mask.zip
# !unzip -qa -d /home/aistudio/data/place2/ /home/aistudio/data/data105124/places2.zip
# # if not os.path.exists("./data/coco_stuff/train_inst"):
# #     os.mkdir("./data/coco_stuff/train_inst")
# # !unzip -qa -d /home/aistudio/data/coco_stuff/train_inst/ /home/aistudio/data/data88531/train_inst.zip

# # !unzip -qa -d /home/aistudio/data/coco_stuff/ /home/aistudio/data/data84834/stuffthingmaps_trainval2017.zip '*train2017/*.png'
# # !mv -T /home/aistudio/data/coco_stuff/train2017 /home/aistudio/data/coco_stuff/train_label

# # %cd /home/aistudio/data/coco_stuff/train_inst/
# # %ls -l |grep "^-"|wc -l
# # %cd /home/aistudio/

# # %cd /home/aistudio/data/coco_stuff/train_label/
# # %ls -l |grep "^-"|wc -l
# # %cd /home/aistudio/

# # %cd /home/aistudio/data/coco_stuff/train_img/
# # %ls -l |grep "^-"|wc -l
# # %cd /home/aistudio/
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory.
# All changes under this directory will be kept even after reset.
# Please clean unnecessary files in time to speed up environment loading.
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required,
# you need to use the persistence path as the following:
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```

    mkdir: cannot create directory ‘/home/aistudio/external-libraries’: File exists
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting beautifulsoup4
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/69/bf/f0f194d3379d3f3347478bd267f754fc68c11cbf2fe302a6ab69447b1417/beautifulsoup4-4.10.0-py3-none-any.whl (97kB)
    [K     |████████████████████████████████| 102kB 26.6MB/s ta 0:00:01
    [?25hCollecting soupsieve>1.2 (from beautifulsoup4)
      Downloading https://mirror.baidu.com/pypi/packages/36/69/d82d04022f02733bf9a72bc3b96332d360c0c5307096d76f6bb7489f7e57/soupsieve-2.2.1-py3-none-any.whl
    Installing collected packages: soupsieve, beautifulsoup4
    Successfully installed beautifulsoup4-4.10.0 soupsieve-2.2.1
    [33mWARNING: Target directory /home/aistudio/external-libraries/soupsieve-2.2.1.dist-info already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/beautifulsoup4-4.10.0.dist-info already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/bs4 already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/soupsieve already exists. Specify --upgrade to force replacement.[0m



```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code,
# so that every time the environment (kernel) starts,
# just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.


```python
#添加设置全局参数

import warnings
warnings.filterwarnings('ignore')

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# set up global parameters
# 修改了syncbatchnorm为batchnorm
# 修改了 dataroot
# batchSize 设为 4
# 调整 vgg loss lambda 为 0.2
# 优化器的参数和学习率直接在下面，不用OPT设置。
d_lr, g_lr, beta1, beta2 = 4e-4, 1e-4, 0., .999

class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        # self.batchSize=1
        self.batchSize=8
        self.beta1=0.5
        self.beta2=0.999
        self.cache_filelist_read=True
        self.cache_filelist_write=True
        self.checkpoints_dir='checkpoints'
        self.coco_no_portraits=False
        self.contain_dontcare_label=True
        self.continue_train=False
        self.crop_size=256
        # self.dataroot='./datasets/cityscapes/'
        self.maskroot='data/mask/center_mask'
        self.dataroot='data/place2/places2'
        # self.dataroot='/home/aistudio/coco_stuff/'
        self.dataset_mode='inpainting'
        self.display_freq=500
        self.display_winsize=256
        self.gan_mode='hinge'
        self.gpu_ids=[]
        self.init_type='xavier'
        self.init_variance=0.02
        self.isTrain=True
        self.label_nc=182
        self.lambda_feat=10.0
        self.lambda_kld=0.05
        # self.lambda_vgg=10.0
        self.lambda_vgg=.2
        self.load_from_opt_file=False
        self.load_size=256
        self.d_lr=0.0002
        self.g_lr=0.0002
        self.model='pix2pix'
        self.nThreads=0
        self.n_layers_D=4
        self.name='Place(noGAN)'
        self.ndf=64
        self.nef=16
        self.netD='multiscale'
        self.netD_subarch='n_layer'
        self.netG='spade'
        self.ngf=64
        self.niter=50
        self.niter_decay=0
        self.norm_D='spectralinstance'
        self.norm_E='spectralinstance'
        # self.norm_G='spectralspadesyncbatch3x3'
        self.norm_G='spectralspadebatch3x3'
        self.num_D=2
        self.num_upsampling_layers='normal'
        self.optimizer='adam'
        self.output_nc=3
        self.phase='train'
        self.preprocess_mode='resize_and_crop'
        self.print_freq=100
        self.save_epoch_freq=5
        self.save_latest_freq=5000
        self.which_epoch='latest'
        self.num_workers=4
        self.current_epoch=0
        self.epoch_num=120
        self.lambda_GAN=0.2
        self.lambda_style=120
        self.lambda_prc=0.05
        self.lambda_hole=6
        self.lambda_valid=1
        self.lambda_l1=1
        self.lambda_tv=0.1

        self.log_dir='./logs'
opt = OPT()
```


```python
#加载数据集
# 加载数据集
from paddle.io import Dataset, DataLoader
# from paddle.vision.transforms import Resize
import paddle.vision.transforms as transforms
import numpy as np
import random
from glob import glob
from PIL import Image


# 处理图片数据：裁切、水平翻转、调整图片数据形状、归一化数据

    # self.maskroot='data/irregular_mask/irregular_mask'
    # self.dataroot='data/celeba_hq/celeba_hq/train'


# 定义Inpaint数据集对象
class InpaintDateset(Dataset):
    def __init__(self, opt):
        super(InpaintDateset, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor()
        ])

        # img_dir = opt.dataroot+'train_img/'
        # _, _, image_list = next(os.walk(img_dir))
        # self.image_list = np.sort(image_list)
        self.opt = opt
        self.img_root = sorted(glob('{:s}/*.jpg'.format(self.opt.dataroot)))
        self.mask_root = sorted(glob('{:s}/*.png'.format(self.opt.maskroot)))
        self.N_mask = len(self.mask_root)
        # inst_dir = opt.dataroot+'/'
        # inst_list = next(os.walk(inst_dir))
        # self.inst_list = np.sort(inst_list)
        print(len(self.img_root))
        print(self.N_mask)

    def __getitem__(self, idx):
        img = Image.open(self.img_root[idx])
        img = img.convert('RGB')
        mask = Image.open(self.mask_root[random.randint(0, self.N_mask - 1)])
        mask = mask.convert('RGB')
        de_img = self.img_transform(img)
        de_mask = self.mask_transform(mask)

        # de_img = data_transform(img, load_size=opt.load_size, is_image=True)
        # de_mask = data_transform(mask, load_size=opt.load_size, is_image=False)

        # 把图片改成masked image

        return de_img, de_mask

    def __len__(self):
        return len(self.img_root)
```


```python
#PC的实现
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision import models
import functools


class VGG16FeatureExtractor(nn.Layer):
    def __init__(self):
        super(VGG16FeatureExtractor,self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])  # torch.size([5,h,w])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():  # getattr()用于返回一个对象属性值
                param.requires_grad = False  # 不需要保存梯度

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2D(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=None)
        self.mask_conv = nn.Conv2D(1, 1, kernel_size,
                                   stride, padding, dilation, groups, bias_attr=False,
                                   weight_attr=nn.initializer.Constant(value=1.0))

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        #mask:black=1,white=0
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        input1=input*mask

        output = self.input_conv(input1)  ##原来是input*mask，我改了


        with paddle.no_grad():
            output_mask = self.mask_conv(mask)

        new_mask = paddle.clip(output_mask, 0, 1)
        return output, new_mask


class PCBActiv(nn.Layer):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=None):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)
        if bn:
            self.bn = bn
        self.norm_layer = nn.BatchNorm2D(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)

        if hasattr(self, 'bn'):
            h = self.norm_layer(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask



```


```python
import paddle
import paddle.nn as nn



# 定义encoder

# define the encoder skip connect
class UnetSkipConnectionEBlock(nn.Layer):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionEBlock,self).__init__()
        downconv = nn.Conv2D(outer_nc,inner_nc,kernel_size=4,stride=2,padding=1)
        conv = nn.Conv2D(outer_nc,inner_nc,kernel_size=5,stride=1,padding=2)
        downrelu = nn.LeakyReLU(0.2,True)
        downnorm = nn.BatchNorm2D(inner_nc)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downconv, downrelu]
            model = down
        else:
            down = [downconv, downrelu, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# 定义残差block
class ResnetBlock(nn.Layer):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Pad2D(dilation, mode='reflect'),
            nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias_attr=False),
            nn.InstanceNorm2D(dim),
            nn.ReLU(True),
            nn.Pad2D(1, mode='reflect'),
            nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias_attr=False),
            nn.InstanceNorm2D(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, res_num=4, use_dropout=False):
        super(Encoder, self).__init__()

        # Unet structure

        self.ec_1 = PCBActiv(input_nc, ngf, bn=False, activ=None, sample='down-7')
        self.ec_2 = PCBActiv(ngf, ngf * 2, sample='down-5')
        self.ec_3 = PCBActiv(ngf * 2, ngf * 4, sample='down-5')
        self.ec_4 = PCBActiv(ngf * 4, ngf * 8, sample='down-3')
        self.ec_5 = PCBActiv(ngf * 8, ngf * 8, sample='down-3')
        self.ec_6 = PCBActiv(ngf * 8, ngf * 8, bn=False, sample='down-3')

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

    def forward(self, input, mask):
        y_1, m_1 = self.ec_1(input, mask)
        y_2, m_2 = self.ec_2(y_1, m_1)
        y_3, m_3 = self.ec_3(y_2, m_2)
        y_4, m_4 = self.ec_4(y_3, m_3)
        y_5, m_5 = self.ec_5(y_4, m_4)
        y_6, _ = self.ec_6(y_5, m_5)

        y_7 = self.middle(y_6)

        return y_1, y_2, y_3, y_4, y_5, y_7


```


```python
import paddle
import paddle.nn as nn
#定义decoder
# define the decoder skip connect
class UnetSkipConnectionDBlock(nn.Layer):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upconv = nn.Conv2DTranspose(inner_nc,outer_nc,kernel_size=4,stride=2,padding=1)
        upnorm = nn.BatchNorm2D(outer_nc)
        if outermost:
            print('using relu,bn,conv')
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)

# define decoder
class Decoder(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False):
        super(Decoder, self).__init__()

        #Unet structure
        self.dc_1 = UnetSkipConnectionDBlock(ngf*8, ngf*8, use_dropout=use_dropout, innermost=True)
        self.dc_2 = UnetSkipConnectionDBlock(ngf*16, ngf*8, use_dropout=use_dropout)
        self.dc_3 = UnetSkipConnectionDBlock(ngf*16, ngf*4, use_dropout=use_dropout)
        self.dc_4 = UnetSkipConnectionDBlock(ngf*8, ngf*2, use_dropout=use_dropout)
        self.dc_5 = UnetSkipConnectionDBlock(ngf*4, ngf, use_dropout=use_dropout)
        self.dc_6 = UnetSkipConnectionDBlock(ngf*2, output_nc, use_dropout=use_dropout, outermost=True)

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.dc_1(input_6)
        y_2 = self.dc_2(paddle.concat([y_1, input_5], 1))
        y_3 = self.dc_3(paddle.concat([y_2, input_4], 1))
        y_4 = self.dc_4(paddle.concat([y_3, input_3], 1))
        y_5 = self.dc_5(paddle.concat([y_4, input_2], 1))
        y_6 = self.dc_6(paddle.concat([y_5, input_1], 1))
        out = y_6

        return out

```


```python
import paddle
import paddle.nn as nn

#定义损失函数
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.shape
    # feat = feat.view(b, ch, h * w)
    feat = paddle.reshape(feat, [b, ch, h*w])
    feat_t = paddle.transpose(feat, perm=[0, 2, 1])
    gram = paddle.bmm(feat, feat_t) / (ch * h * w) #torch.bmm(a,b)，a和b进行矩阵的乘法，且a，b都是三维的tensor，必须满足a是(b,h,w)b是(b,w,h)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = paddle.mean(paddle.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        paddle.mean(paddle.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Layer):
    def __init__(self, extractor):
        super(InpaintingLoss,self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output   #I_comp   mask是缺失区域为0

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)
        # loss_dict['l1'] = self.l1(output,gt)

        if output.shape[1] == 3:  #如果output的通道是三通道
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:  #torch.cat是将两个张量拼接在一起，cat是concatnate的意思，后面的数字代表是第几列开始拼接
            feat_output_comp = self.extractor(paddle.concat([output_comp]*3, 1))  #拼接第一列，相当于将b*1*h*w的tensor变成b*3*h*w
            feat_output = self.extractor(paddle.concat([output]*3, 1))
            feat_gt = self.extractor(paddle.concat([gt]*3, 1))
        else:
            raise ValueError('only gray an') #程序抛出异常

        loss_dict['prc'] = 0.0  #感知损失
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)
        # loss_dict['tv'] = total_variation_loss(output)

        return loss_dict




```


```python

```


```python
import paddle
import random
from collections import OrderedDict
import numpy as np

class BASE():
    def __init__(self, opt):
        super(BASE,self).__init__()

        self.opt = opt
        self.istrain=opt.isTrain
        self.criterion=InpaintingLoss(VGG16FeatureExtractor())

        self.net_EN = Encoder(3,64)
        self.net_DE = Decoder(64,3)


        if self.istrain:

            self.net_EN.train()
            self.net_DE.train()

        self.opt_en = paddle.optimizer.Adam(learning_rate=opt.g_lr, beta1=opt.beta1, beta2=opt.beta2, parameters=self.net_EN.parameters())
        self.opt_de = paddle.optimizer.Adam(learning_rate=opt.g_lr, beta1=opt.beta1, beta2=opt.beta2, parameters=self.net_DE.parameters())




        if self.istrain:
            if opt.continue_train:
                pass

    def mask_process(self, mask):
        mask = mask[0][0]
        mask = paddle.unsqueeze(mask,0)
        mask = paddle.unsqueeze(mask,1)
        return mask

    def set_input(self, input_De, mask):

        self.Gt_DE = input_De
        self.input_DE = input_De
        self.mask_global = (1-self.mask_process(mask))#black=1,white=0
        self.Gt_Local = input_De
        # define local area which send to the local discriminator
        self.crop_x = random.randint(0, 191)
        self.crop_y = random.randint(0, 191)
        self.Gt_Local = self.Gt_Local[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        self.ex_mask = paddle.expand(self.mask_global,[self.mask_global.shape[0], 3, self.mask_global.shape[2],
                                               self.mask_global.shape[3]])


        # Do not set the mask regions as 0
        self.input_DE = self.input_DE*self.mask_global

    def forward(self):

        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.net_EN(
            self.input_DE,self.mask_global)
        De_in = [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6]
        self.fake_out = self.net_DE(De_in[0], De_in[1], De_in[2], De_in[3], De_in[4], De_in[5])


    def backward_G(self):
        # First, The generator should fake the discriminator
        real_AB = self.Gt_DE
        fake_AB = self.fake_out
        real_local = self.Gt_Local
        fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]

        self.loss_dict=self.criterion(self.input_DE,self.mask_global,self.fake_out, self.Gt_DE)


        self.loss_G = self.loss_dict['style'] * self.opt.lambda_style + \
                      self.loss_dict['hole'] * self.opt.lambda_hole + self.loss_dict['valid'] * self.opt.lambda_valid + \
                      self.loss_dict['prc'] * self.opt.lambda_prc + self.loss_dict['tv'] * self.opt.lambda_tv


        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = requires_grad

    def optimize_parameters(self):
        self.forward()

        self.opt_en.clear_grad()
        self.opt_de.clear_grad()

        self.backward_G()
        self.opt_en.step()
        self.opt_de.step()

    def get_current_errors(self):
        # print('l1:%.4f,style:%.4f,tv:%.4f,prc:%.4f' % (
        #     #self.loss_G_GAN.item() * self.opt.lambda_GAN,
        #     self.loss_dict['hole'] * self.opt.lambda_hole + self.loss_dict['valid'] * self.opt.lambda_valid,
        #     self.loss_dict['style'] * self.opt.lambda_style, self.loss_dict['tv'] * self.opt.lambda_tv,
        #     self.loss_dict['prc'] * self.opt.lambda_prc))
        # show the current loss
        # return OrderedDict([#('G_GAN', self.loss_G_GAN),
        #                     ('G_L1', self.loss_G)
        #                     #('D', self.loss_D_fake),
        #                     #('F', self.loss_F_fake)
        #                     ])
        print(self.loss_G)
    # You can also see the Tensorborad
    def get_current_visuals(self):
        input_image = (self.input_DE + 1) / 2.0
        # input_image2 = (self.input_DE2.data.cpu()+1) / 2.0
        # input_image3 = (self.input_DE3.data.cpu() + 1) / 2.0
        fake_image = (self.fake_out + 1) / 2.0
        real_gt = (self.Gt_DE + 1) / 2.0

        return input_image[0].unsqueeze(0), fake_image[0].unsqueeze(0), real_gt[0].unsqueeze(0)

    def save_epoch(self,epoch):
        # paddle.save(self.netD.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_d.pdparams")
        # paddle.save(self.opt_d.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_d.pdopt")
        paddle.save(self.net_EN.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_en.pdparams")
        paddle.save(self.opt_en.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_en.pdopt")
        paddle.save(self.net_DE.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_de.pdparams")
        paddle.save(self.opt_de.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_de.pdopt")
        # paddle.save(self.netF.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_f.pdparams")
        # paddle.save(self.opt_f.state_dict(), self.opt.checkpoints_dir + "model/" + str(epoch) + "_f.pdopt")





```


```python
import time
import os
import paddle
from paddle.io import Dataset, DataLoader
from PIL import Image
import numpy as np

if __name__ == "__main__":

    opt = OPT()
    dataset = InpaintDateset(opt)
    print(dataset)
    # dataloader = DataLoader(dataset, batch_size=opt.load_size, shuffle=True, num_workers=2)

    loader = DataLoader(dataset,
                        batch_size=opt.batchSize,
                        shuffle=True,
                        drop_last=True,
                        num_workers=opt.num_workers)

    model = BASE(opt)
    total_steps = 0


    if opt.current_epoch > 0:
        print('读取存储的模型权重、优化器参数...')

        en_statedict_model = paddle.load(opt.checkpoints_dir+"model/"+str(opt.current_epoch)+"_en.pdparams")
        model.net_EN.set_state_dict(en_statedict_model)
        de_statedict_model = paddle.load(opt.checkpoints_dir+"model/"+str(opt.current_epoch)+"_de.pdparams")
        model.net_DE.set_state_dict(de_statedict_model)
        en_statedict_opt = paddle.load(opt.checkpoints_dir+"model/"+str(opt.current_epoch)+"_en.pdopt")
        model.opt_en.set_state_dict(en_statedict_opt)
        de_statedict_opt = paddle.load(opt.checkpoints_dir+"model/"+str(opt.current_epoch)+"_de.pdopt")
        model.opt_de.set_state_dict(de_statedict_opt)


    # Start Training
    for epoch in range(opt.current_epoch+1,opt.epoch_num+1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for detail, mask in loader():
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(detail, mask)
            # display the training loss
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                print('iters: %d iteration time: %.10fsec' % (total_steps, t))
            # 定时存盘
        if epoch % opt.save_epoch_freq == 0:
            model.save_epoch(epoch)
            print('第['+str(epoch)+']轮模型保存。')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.current_epoch+1+opt.epoch_num, time.time() - epoch_start_time))


```

    10000
    1000
    <__main__.InpaintDateset object at 0x7fa49507fad0>
    using relu,bn,conv



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
