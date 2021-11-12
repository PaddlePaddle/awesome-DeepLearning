# Wide_Resnet
简体中文

   * [Wide Resnet](#Wide_Resnet)
      * [一、简介](#一简介)
      * [二、复现精度](#二复现精度)
      * [三、数据集](#三数据集)
      * [四、环境依赖](#四环境依赖)
      * [五、快速开始](#五快速开始)
         * [step1: 训练](#step1-训练)
         * [step2: 评估](#step2-评估)
      * [六、代码结构与详细说明](#六代码结构与详细说明)
         * [6.1 代码结构](#61-代码结构)
         * [6.2 参数说明](#62-参数说明)
         * [6.3 训练流程](#63-训练流程)
            * [单机训练](#单机训练)
            * [多机训练](#多机训练)
            * [训练输出](#训练输出)
         * [6.4 评估流程](#64-评估流程)
      * [七、模型信息](#七模型信息)

## 一、简介

本项目基于paddlepaddle框架复现Wide Resnet，他是resnet的一种变体，主要区别在于对resnet的shortcut进行了改进，使用更“宽”的卷积以及加上了dropout层。


**论文:**
- [1]  Zagoruyko S ,  Komodakis N . Wide Residual Networks[J].  2016.<br>
- 链接：[Wide Residual Networks](https://arxiv.org/abs/1605.07146)

**参考项目：**
- [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- [https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py)


## 二、复现精度

>该列指标在cifar10的测试集测试

train from scratch细节：


| |epoch|opt|batch_size|dataset|memory|card|precision|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|1|400|SGD|128|CIFAR10|16G|1|0.9660|

**模型下载**
模型地址：[aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104172)


## 三、数据集

[CIFAR10数据集](http://www.cs.toronto.edu/~kriz/cifar.html)。

- 数据集大小：
  - 训练集：50000张
  - 测试集：10000张
  - 尺寸：32 * 32
- 数据格式：分类数据集

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0

## 五、快速开始

### step1: clone

```bash
# clone this repo
git clone https://github.com/PaddlePaddle/Contrib.git
cd wide_resnet
export PYTHONPATH=./
```
**安装依赖**
```bash
python3 -m pip install -r requirements.txt
```

### step2: 训练
```bash
python3 train.py
```
如果你想分布式训练并使用多卡：
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py
```

此时的输出为：
```
Epoch 0: PiecewiseDecay set learning rate to 0.05.
iter:0  loss:2.4832
iter:10  loss:2.3544
iter:20  loss:2.3087
iter:30  loss:2.2509
iter:40  loss:2.2450
```

### step3: 测试
```bash
python3 eval.py
```
此时的输出为：
```
acc:9660 total:10000 ratio:0.966
```

## 六、代码结构与详细说明

### 6.1 代码结构

```
│  wide_resnet.py                 # 模型文件
│  eval.py                        # 评估
│  README.md                      # 英文readme
│  README_cn.md                   # 中文readme
│  requirement.txt                # 依赖
│  train.py                       # 训练
```

### 6.2 参数说明

无


### 6.3 训练流程

#### 单机训练
```bash
python3 train.py
```

#### 多机训练
```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3' train.py
```

此时，程序会将每个进程的输出log导入到`./debug`路径下：
```
.
├── debug
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
├── README.md
└── train.py
```

#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。
```text
Epoch 0: PiecewiseDecay set learning rate to 0.05.
iter:0  loss:2.4832
iter:10  loss:2.3544
iter:20  loss:2.3087
iter:30  loss:2.2509
iter:40  loss:2.2450
```

### 6.4 评估流程

```bash
python3 eval.py
```

此时的输出为：
```
acc:9660 total:10000 ratio:0.966
```
## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 徐铭远|
| 时间 | 2021.08 |
| 框架版本 | >=Paddle 2.0.2|
| 应用场景 | 图像分类 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/drive/folders/1Xf5NsmxseygbDKYLBgSZcnvy4fRq6ZzY?usp=sharing)  |



```python
# 以下为在aistudio上直接运行
```


```python
# 训练
!python3 train.py
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    W0808 16:41:54.148313 32483 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0808 16:41:54.152312 32483 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    Epoch 0: PiecewiseDecay set learning rate to 0.05.
    iter:0  loss:2.4279
    iter:10  loss:2.3434
    ^C



```python
# 评估
!python3 eval.py
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    W0808 16:37:21.490298 32096 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0808 16:37:21.494925 32096 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    acc:9660 total:10000 ratio:0.966
