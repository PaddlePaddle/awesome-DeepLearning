# SID-Paddle

## 一、简介
【飞桨论文复现挑战赛】
本项目是用百度飞桨框架paddlepaddle复现：Learning to See in the Dark in CVPR 2018, by [Chen Chen](http://cchen156.github.io/), [Qifeng Chen](http://cqf.io/), [Jia Xu](http://pages.cs.wisc.edu/~jiaxu/), and [Vladlen Koltun](http://vladlen.info/).

**原代码地址：**[Learning-to-See-in-the-Dark](http://cchen156.github.io/SID.html)

**论文：**[Learning to See in the Dark ](http://cchen156.github.io/paper/18CVPR_SID.pdf)

本代码包含了原论文的默认配置下的训练和测试代码。

## 二、复现精度

| 指标 | 原论文 | 原代码精度 | 复现精度 |
| --- | --- | --- | --- |
| PSNR | 28.88 | 28.96 | 28.82 |
| SSIM | 0.787 | 0.785 | 0.787 |

## 三、数据集
使用的数据集为：[SID-Sony](https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA#list/path=%2F)
注：如果只使用Sony数据集，只需要下载Sony开头的文件，下载结束后，使用"cat SonyPart* > Sony.zip"得到数据集压缩包。

- 请注意！根据原作者在原代码README中的描述，在进行定量评估时(运行eval.py之前)，需要删除10034，10045和10172这几组图片，因为这些图片对应的长短曝光图片有些许位移，会影响定量评估的结果。

- 数据集大小：包含2697个短曝光图像和231个长曝光图像，每一个短曝光图像都有一个对应的长曝光图像(ground truth)，多个短曝光图像可能对应同一个长曝光图像。
  - 训练集：1865短曝光+161长曝光
  - 验证集：234短曝光+20长曝光
  - 测试集：598短曝光+50长曝光
- 数据集格式：本数据集使用RAW格式图片，jpg或者png等格式不支持本网络。

## 四、环境依赖
- 硬件：本项目在Paddle AI Studio平台 4卡Tesla V100, 128G显存上运行，训练4000 epoch需要约16小时。根据原作者称，显存大小最小需要64G
- 框架：
  - Paddlepaddle >= 2.0.0
  - rawpy
  - scipy == 1.1.0

## 五、快速开始

### step1: clone

```bash
# clone this repo
git clone git://github.com/WangChen0902/SID-Paddle.git
```

### step2: 下载数据

将上文提到的数据集放到本项目data目录下，目录格式：SID-Paddle/data/Sony

### step3: 训练

```bash
python train_Sony_paddle.py  # 单卡
python -m paddle.distributed.launch train_Sony_paddle.py  # 单机多卡
```


### step3: 测试

```bash
python test_Sony_paddle.py
```

### step4: 评估

```bash
python eval.py
```

## 六、代码结构与详细说明

### 6.1 代码结构

```
├── checkpoint  # 存放模型文件的路径
├── data  # 存放数据集的路径
├── result  # 存放程序输出的路径
├── utils  # 工具类
│   ├── PSNR.py
│   └── SSIM.py
├── eval.py  # 评估程序，计算PSNR/SSIM
├── README.md
├── run.sh  # AI Studio 单机多卡训练运行脚本
├── run_test.sh  # 单机单卡测试脚本
├── test_Sony_paddle.py  # 测试程序
└── train_Sony_paddle.py  # 训练程序
```

### 6.2 参数说明

|  参数   | 默认值  | 说明 |
|  ----  |  ----  |  ----  |
| start_epoch | 0 | 起始epoch值 |
| num_epoches | 4001 | epoch次数 |
| patch_size | 512 | 用于训练的图片大小 |
| save_freq | 200 | 保存模型的频率 |
| learning_rate | 1e-4 | 学习率 |
| DEBUG | 0 | 是否开启调试 |
| data_prefix | './data/' | 数据集路径 |
| output_prefix | './result/' | 输出路径 |
| checkpoint_load_dir | './checkpoint/' | 读取模型的路径 |
| last_epoch | 4000 | 测试时读取哪一轮的模型 |

## 七、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | Wangchen0902 |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 图像增强 |
| 支持硬件 | GPU>=64G |
| 下载链接 | [预训练模型](https://pan.baidu.com/s/1FF1K3lbsTT24tY91qIUZWg) 提取码：6hbx |
| 下载链接 | [训练日志](https://pan.baidu.com/s/1q7HvQVRwZxoGQHon_tO2YA) 提取码：brfz |
| 在线运行 | [SID-Paddle](https://aistudio.baidu.com/aistudio/projectdetail/2275443) |
