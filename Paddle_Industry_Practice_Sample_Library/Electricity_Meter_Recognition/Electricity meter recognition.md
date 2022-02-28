# 多类别电表读数识别
## 1 项目背景
我国电力行业发展迅速，电表作为测电设备经历了普通电表、预付费电表和智能电表三个阶段的发展，虽然智能电表具有通信功能，但一方面环境和设备使得智能电表具有不稳定性，另一方面非智能电表仍然无法实现自动采集，人工抄表有时往往不可取代。采集到的大量电表图片如果能够借助人工智能技术批量检测和识别，将会大幅提升效率和精度。

在本系列项目中，我们使用Paddle工具库实现一个OCR垂类场景。原始数据集是一系列电度表的照片，类型较多，需要完成电表的读数识别，对于有编号的电表，还要完成其编号的识别。

![](https://ai-studio-static-online.cdn.bcebos.com/26b64bd400e44096af03b9ca90539a706b77e8e21c3441b696e6aaaf850f00aa)


本项目难点包括：

* 在数据方面，电表种类多、数据少，拍摄角度多样且部分数据反光严重。
* 电表数据没有开源数据集，如何从零标注数据应当选择何种标注软件能够最快速度构建数据集？
* 在技术路线选择也面临多方面的问题，例如是通过文字检测来反向微调，还是通过目标检测从零训练？


最终的项目方案为：

使用飞桨文字识别开发套件PaddleOCR，完成PP-OCR模型完成微调与优化，由于其检测部分基于DB的分割方法实现，对于数据中的倾斜问题能够良好解决。PP-OCR模型经过大量实验，其泛化性也足以支撑复杂垂类场景下的效果。



## 2 安装说明

环境要求
* PaddlePaddle >= 2.1.0
* 3.5 <= Python < 3.9
* PaddleOCR >= 2.1

```python
# 克隆项目
!git clone https://gitee.com/paddlepaddle/PaddleOCR.git
```


```python
# 安装ppocr
!pip install fasttext==0.8.3
!pip install paddleocr --no-deps -r requirements.txt
```


```python
%cd PaddleOCR/
```


## 3 数据集简介

> 注：数据集稍后公开，尽请期待

首先，我们来简单看一下数据集的情况。总的来说，这个场景面临几个比较大的问题：
- 电表类型较多。相比之下，现有数据量（500张）可能不够。
- 照片角度倾斜较厉害。这个比较好理解，有些电表可能不具备正面拍照条件，有不少图片是从下往上、甚至从左下往右上拍的。
- 反光严重。反光问题对定位目标框以及识别数字可能都会产生影响。
- 表号是点阵数字，不易识别。这个问题是标注的时候发现的，有的标注，PPOCRLabel自动识别的四点检测定位其实已经非常准了，但里面的数字识别效果却很离谱。
- 对检测框精准度要求非常高。电表显示读数的地方附近一般不是空白，往往有单位、字符或是小数点上的读数等，如果检测框没框准，就会把其它可识别项纳进来，如果也是数字，就算加了后处理也处理不掉。

下面，读者可以通过这几张典型图片，初步感受下数据集的基本情况。


![](https://ai-studio-static-online.cdn.bcebos.com/ea1b247d86a241f5876fc37902f9892a6fa9bacf639e40b8b8a9c9e0837ef8bf)
![](https://ai-studio-static-online.cdn.bcebos.com/b1ae776ddd3443d4ad1bf74a3bc4ba3afe0dcc8e32754c76a584262305c004f9)
![](https://ai-studio-static-online.cdn.bcebos.com/2286772557d1410b931d702990dea5f6fc470996013b4064aa3244b532c677c6)

## 4 数据标注

在数据标注工具上，使用PPOCRLabel作为实现半自动标注，内嵌PP-OCR模型，一键实现机器自动标注，且具有便捷的修改体验。支持四点框、矩形框标注模式，导出格式可直接用于PaddleOCR训练。


标注文件格式如下所示：

```
" 图像文件名                    json.dumps编码的图像标注信息"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]

```


## 5 模型选型
PaddleOCR包含丰富的文本检测、文本识别以及端到端算法。在PaddleOCR的全景图中，我们可以看到PaddleOCR支持的文本检测算法。

<div align="center">
<img src="https://github.com/paddlepaddle/PaddleOCR/raw/release/2.4/doc/overview.png"  width = "800" height = "500" />
</div>




在标注数据的基础上，基于通用的文本检测算法finetune，我们就可以训练一个能将电表识别中的多余文本框自动去除，只留下目标的电表读数、编号的电表文本检测模型。

![](https://ai-studio-static-online.cdn.bcebos.com/4ad67295c3a94f18a7cfde875fb25b9834a7fda08ba346858a2d56651c1f4ba8)

明确了目标，我们开始下一步的操作。

## 6 检测模型训练
为节省训练时间，这里提供了一个效果不错的预训练模型以及配置文件，读者可以选择基于预训练模型finetune或是从头训练。

在AIStudio训练，一定要注意几个重点！
- 用至尊版！因为原图分辨率太大，目标框相对其实很小，所以输入模型的size太小训练效果不好，而size设大自然需要更多显存
- `use_shared_memory`设置为`False`
- `batch_size_per_card`不能设置太大，因为输入size比较大

后面两个tricks如果不照做，训练会闪退。

本项目使用 `configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml` 配置文件。文件中更改点如下所示：

```yaml
Global:
  debug: false
  use_gpu: true
  epoch_num: 1200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/det_dianbiao_v3
  save_epoch_step: 1200
  eval_batch_step:
  - 0
  - 100
  cal_metric_during_train: false
  pretrained_model: my_exps/student.pdparams
  checkpoints: null
  save_inference_dir: null
  use_visualdl: false
  infer_img: M2021/test.jpg
  save_res_path: ./output/det_db/predicts_db.txt
Architecture:
  model_type: det
  algorithm: DB
  Transform: null
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: large
    disable_se: true
  Neck:
    name: DBFPN
    out_channels: 96
  Head:
    name: DBHead
    k: 50
Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0001 # 调小学习率
    warmup_epoch: 2 
  regularizer:
    name: L2
    factor: 0
PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5
Metric:
  name: DetMetric
  main_indicator: hmean
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
    - M2021/M2021_label_train.txt  # 数据标注文件路径
    ratio_list:
    - 1.0
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - CopyPaste: null  # 使用数据增强
    - IaaAugment:
        augmenter_args:
        - type: Fliplr
          args:
            p: 0.5
        - type: Affine
          args:
            rotate:
            - -10
            - 10
        - type: Resize
          args:
            size:
            - 0.5
            - 3
    - EastRandomCropData:
        size:
        - 1600  # 增大图像分辨率大小
        - 1600
        max_tries: 50
        keep_ratio: true
    - MakeBorderMap:
        shrink_ratio: 0.4
        thresh_min: 0.3
        thresh_max: 0.7
    - MakeShrinkMap:
        shrink_ratio: 0.4
        min_text_size: 8
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - threshold_map
        - threshold_mask
        - shrink_map
        - shrink_mask
  loader:
    shuffle: true
    drop_last: false
    batch_size_per_card: 4 # 重点！
    num_workers: 4
    use_shared_memory: False # 重点！
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
    - M2021/M2021_label_eval.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - DetResizeForTest:
        limit_side_len: 1280
        limit_type: min
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - shape
        - polys
        - ignore_tags
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
    use_shared_memory: False # 重点！
profiler_options: null
```

按照上方配置文件更改后运行下方代码


```python
# 从头开始训练
!python tools/train.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml
```

## 7 模型评估与推理

通过上述代码训练好模型后，运行 `tools/eval.py`, 指定配置文件和模型参数即可评估效果。

> 由于我们只会开源部分数据集，全量数据的训练模型可在 `my_exp` 文件中找到，运行下方代码可直接测试调优后模型结果。

实验结果如下表所示：

| 模型 | 实验 | 指标 |
| -------- | -------- | -------- |
| PP-OCRv2     | 仅调整lr     | H-Means=0.3     |
| PP-OCRv2     | 调整lr+增大输入分辨率     | H-Means=0.65     |
| PP-OCRv2     | 调整lr+增大输入分辨率+CopyPaste    | H-Means=0.85     |


```python
# 提供的预训练模型和配置文件（供参考，直接用不该上面两个注意点，训练会报错）
!tar -xvf ../my_exps.tar -C ./
```


```python
# 也可以查看下提供的模型训练效果
!python tools/eval.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml  -o Global.checkpoints="my_exps/det_dianbiao_size1600_copypaste/best_accuracy"
```


```python
!python tools/infer_det.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml -o Global.infer_img="./M2021/test.jpg" -o Global.checkpoints="my_exps/det_dianbiao_size1600_copypaste/best_accuracy"
```

![](https://ai-studio-static-online.cdn.bcebos.com/102212c5eb484c1199eddcf7fac7b3c9c88508c05eec4118b1ab61cc9328b798)

效果非常棒！接下来，就是串接检测模型和识别模型了。

## 8 模型导出和串接
这里用了个比较取巧的方式，先将模型导出，然后把`whl`下预测用的检测模型用新训练的模型直接替换掉，就可以看到finetune后的检测效果了！


```python
# 模型导出
!python tools/export_model.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml -o Global.pretrained_model=./my_exps/det_dianbiao_size1600_copypaste/best_accuracy Global.save_inference_dir=./inference/det_db
```


```python
from paddleocr import PaddleOCR, draw_ocr
# 模型路径下必须含有model和params文件
ocr = PaddleOCR(det_model_dir='./inference/det_db', 
                use_angle_cls=True)
img_path = './M2021/test.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)

```python
# 显示结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

![](https://ai-studio-static-online.cdn.bcebos.com/1af711fced7b46ac9d47689e9dc657a3af3f2341c6a144b5af6ac1f8a5ad50d7)

如果您想要进一步优化识别结果，可以通过一下两种思路：

1. 重新训练识别模型
   
    * 通过 `导出识别数据` 功能在PPOCRLabel中导出识别数据：包含已经裁切好的识别图片与label
    
    * 如果您的真实数据量太小，使用Textrenderer、StyleText等造数据工具，制造合成数据（可能需要提供字体文件等）。
    
    * 将数据按照[识别模型训练文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/recognition.md)整理数据后启动训练，通过调整学习率、调整相应的合成与真实数据比例（保证每个batch中真实：合成=10：1左右）等操作优化识别模型。

2. 通过后处理解决，包括调整阈值、将非数字内容处理掉等


> 如果您对本项目以及PaddleOCR应用有更深入的需求，欢迎扫码加群交流：
> <div align="center">
> <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/joinus.PNG"  width = "250" height = "250" />
> </div>


## 番外篇：基于目标检测方案的探索

工业场景中对于文字的检测也可以算是目标的一种，因此我们也探索了通用目标检测的方法在该场景中的效果。

整体方案的流程首先将PPOCRLabel的标注文件格式转换为VOC格式，然后训练YOLOv3模型进行文本检测。
具体代码可参考 [PPOCR+PPDET电表读数和编号识别](https://aistudio.baidu.com/aistudio/projectdetail/2803693?contributionType=1)。

最终预测效果如下：

![](https://ai-studio-static-online.cdn.bcebos.com/5542cac177ea4ff483761092855911a611e35aa099134a049a7e2530ec279c15)
![](https://ai-studio-static-online.cdn.bcebos.com/feb8862446584b56ba358ef445734fc726ac23f445284c2fb0fd63bee3f0c9d5)
![](https://ai-studio-static-online.cdn.bcebos.com/7482c71bda9a445e9a6809e66ff2cec3ff2432e2b6ce4506986f3116b10fcbdd)

从上面的预测结果看来，我们发现直接用矩形框检测也存在问题。由于输入图片会存在歪斜，导致矩形框可能会框住多余的文字，进而影响文字识别效果。
