# 	Hybrid Task Cascade for Instance Segmentation

![](https://ai-studio-static-online.cdn.bcebos.com/66df6d440dd7436da1135b665898c696e921d665f7f4423ea20f00bc56fbeb62)


## 一、简介

本项目基于paddledetection框架复现HTC。HTC是一种目标检测实例分割网络，在 cascade rcnn 基础上修改 cascade head（加入mask预测部分，mask之间加入信息传递），并增加分支利用语义分割信息提供空间上下文信息。

 >"级联是一种比较经典的结构，在很多任务中都有用到，比如物体检测中的 CC-Net，Cascade R-CNN，语义分割中的 Deep Layer Cascade 等等。然而将这种结构或者思想引入到实例分割中并不是一件直接而容易的事情，如果直接将 Mask R-CNN 和 Cascade R-CNN 结合起来，获得的提升是有限的，因此需要更多地探索检测和分割任务的关联。
在本篇论文中提出了一种新的实例分割框架，设计了多任务多阶段的混合级联结构，并且融合了一个语义分割的分支来增强 spatial context。这种框架取得了明显优于 Mask R-CNN 和 Cascade Mask R-CNN 的结果。"
——[知乎专栏《实例分割的进阶三级跳：从 Mask R-CNN 到 Hybrid Task Cascade》](https://zhuanlan.zhihu.com/p/57629509)

**论文:**
- [1] K. Chen et al., “Hybrid Task Cascade for Instance Segmentation,” ArXiv190107518 Cs, Apr. 2019, Accessed: Aug. 31, 2021. [Online]. Available: http://arxiv.org/abs/1901.0751 <br>

**参考项目：**
- [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2253839](https://aistudio.baidu.com/aistudio/projectdetail/2253839)
- 脚本任务：[https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473)

**repo:**
- [github](https://github.com/laihuihui/htc)
- [gitee](https://gitee.com/tomatoandtomato/htc)

## 二、复现精度

|  model   | Style  | box AP  | mask AP  |
|  ----  | ----  | ----  | ----  |
| htc-R-50-FPN(official)  | pytorch | 42.3 | 37.4 |
| **htc-R-50-FPN(mine)**  | Paddlepaddle | **42.6** | **37.9** |

**权重及日志下载**
权重地址：[百度网盘](https://pan.baidu.com/s/1fThnatGEWrfFm3Q1fagBjQ) (提取码：yc1r )


```python
# 准备代码
%cd /home/aistudio/work/
!git clone https://gitee.com/tomatoandtomato/htc.git
%cd htc
```

    /home/aistudio/work
    Cloning into 'htc'...
    remote: Enumerating objects: 684, done.[K
    remote: Counting objects: 100% (684/684), done.[K
    remote: Compressing objects: 100% (468/468), done.[K
    Receiving objects:  63% (437/684), 50.58 MiB | 4.15 MiB/s  

## 三、数据集

[COCO 2017](https://cocodataset.org/#download) + [stuffthingmaps_trainval2017](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)

- 数据集大小：
  - 训练集：118287张
  - 验证集：5000张
- 数据格式：图片


```python
# 准备数据集
%cd /home/aistudio/work/htc/dataset/coco
!cp /home/aistudio/data/data97273/annotations_trainval2017.zip ./
!cp /home/aistudio/data/data97273/val2017.zip ./
!cp /home/aistudio/data/data97273/train2017.zip ./
!unzip -q annotations_trainval2017.zip
!unzip -q val2017.zip
!unzip -q train2017.zip
!rm annotations_trainval2017.zip
!rm val2017.zip
!rm train2017.zip
# stuffthingmaps
!mkdir stuffthingmaps
%cd stuffthingmaps
!cp /home/aistudio/data/data103772/stuffthingmaps_trainval2017.zip ./
!unzip -q stuffthingmaps_trainval2017.zip
!rm stuffthingmaps_trainval2017.zip
```


```python
# 安装相关依赖
%cd /home/aistudio/work/htc/
!pip install -r requirements.txt
```

## 四、训练


```python
# 训练
%cd /home/aistudio/work/htc/

# 只训练
# !python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml

# 训练时评估
!python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml --eval
```

## 五、评估


```python
# 评估
%cd /home/aistudio/work/htc/

# 使用预训练模型评估
!mkdir checkpoints
!cp /home/aistudio/data/data103772/model_final.pdparams checkpoints/
!cp /home/aistudio/data/data103772/model_final.pdopt checkpoints/
!python tools/eval.py -c configs/htc/htc_r50_fpn_1x_coco.yml -o weights=checkpoints/model_final

# 用训练得到的最终模型进行评估
# !python tools/eval.py -c configs/htc/htc_r50_fpn_1x_coco.yml -o weights=output/htc_r50_fpn_1x_coco/model_final.pdparams
```
