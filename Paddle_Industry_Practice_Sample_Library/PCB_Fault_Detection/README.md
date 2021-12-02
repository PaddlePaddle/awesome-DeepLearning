# 基于PaddleDetection的PCB瑕疵检测


## 项目说明

随着电子制造的发展，电子产品趋向于多功能，智能化和小型化。作为电子产品的重要精密部件，PCB电路板的质量直接影响产品的性能。因此，质量控制尤为重要。利用人工智能中计算机视觉技术，可以保障PCB板的生产质量。


本项目AI Studio版本请参考：[https://aistudio.baidu.com/aistudio/projectdetail/2504869](https://aistudio.baidu.com/aistudio/projectdetail/2504869)


## 数据准备

印刷电路板（PCB）瑕疵数据集：[数据下载链接](http://robotics.pkusz.edu.cn/resources/dataset/)，是一个公共的合成PCB数据集，由北京大学发布，其中包含1386张图像以及6种缺陷（缺失孔，鼠咬伤，开路，短路，杂散，伪铜），用于检测、分类和配准任务。我们选取了其中适用于检测任务的693张图像，随机选择593张图像作为训练集，100张图像作为验证集。AI Studio上有本项目用到的数据：[https://aistudio.baidu.com/aistudio/datasetdetail/52914](https://aistudio.baidu.com/aistudio/datasetdetail/52914)。


## 代码和环境准备

本项目基于PaddleDetection 2.2版本实现，通过下面的命令下载：

```
git clone https://gitee.com/paddlepaddle/PaddleDetection.git
```

安装PaddleDetection依赖库：
```
cd PaddleDetection/
pip install -r requirements.txt
pip install pycocotools
```

## 数据集分析

在调整配置之前，请首先对数据有一个大概的了解。由于是个小数据，这里只简单分析了几项跟配置息息相关的内容，包括每个种类样本个数、每张图像上平均有几个目标、目标框长宽比分布、目标框占图像比例分布等。

运行dataset_analysis.py：
```
python dataset_analysis.py
```

## 模型训练

本项目为大家提供训练配置文件PCB_faster_rcnn_r50_fpn_3x_coco.yml，使用前请先根据自己的数据存放路径进行目录修改。

通过运行下面的脚本启动训练，--eval参数指定在训练过程中进行评估：
```
python3.7 -u tools/train.py -c PCB_faster_rcnn_r50_fpn_3x_coco.yml --eval
```

## 模型评估

模型训练好后，在验证集上进行评估，此处需要指定评估模型路径-o weights=models/PCB_faster_rcnn_r50_fpn_3x_coco/best_model.pdparams：
```
python -u tools/eval.py -c PCB_faster_rcnn_r50_fpn_3x_coco.yml \
 -o weights=models/PCB_faster_rcnn_r50_fpn_3x_coco/best_model.pdparams use_gpu=True
            
```

## 模型预测

用训练出来的模型在一个PCB图像上进行测试。测试结果保存在output文件夹中，--infer_img=../PCB_DATASET/images/04_missing_hole_10.jpg指定了被推理图像路径：
```
python -u tools/infer.py -c PCB_faster_rcnn_r50_fpn_3x_coco.yml \
                --infer_img=../PCB_DATASET/images/04_missing_hole_10.jpg \
                -o weights=models/PCB_faster_rcnn_r50_fpn_3x_coco/best_model.pdparams use_gpu=True
```


