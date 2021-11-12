# 前言
人工智能创新应用大赛——飞桨开源框架前沿模型复现专题赛，使用Paddle复现Real-Time-High-Resolution-Background-Matting论文。

github： https://github.com/zackzhao1/BackgroundMattingV2-paddle

aistudio： https://aistudio.baidu.com/aistudio/projectdetail/2467759

依赖环境：
paddlepaddle-gpu2.1.2
python3.7

# 论文简介
该方法中将整个pipeline划分为两个部分：base和refine部分，前一个部分在缩小分辨率的输入下生成粗略的结果输出，其主要用于提供大体的区域位置定位（coarse predcition）。后一个网络在该基础上通过path selection选取固定数量的path（这些区域主要趋向于选择头发/手等难分区域）进行refine，之后将path更新之后的结果填充回原来的结果，从而得到其在高分辨率下的matting结果。
![](https://ai-studio-static-online.cdn.bcebos.com/e2d074ae47b441f9a513f390062ced915a17f53d920740ef9bef8cccc62c54db)



# 复现
![](https://ai-studio-static-online.cdn.bcebos.com/ec460c8e1ecc47c4937b0d25fd25cce21efb75187e504f6490a69104905c7e0d)

模型下载 链接：https://pan.baidu.com/s/1WfpzLcjaDJPXYSrzPWvsyQ 提取码：nsfy


# 训练

* stage1：使用VideoMatte240K数据集做预训练，提升模型鲁棒性。

注：由于预训练耗时较长，提供了训练好得模型，方便在自己的数据上微调，模型为stage1.pdparams。

* stage2：使用Distinctions646数据集做微调，提升模型细节表现。

注：此时模型最好精度为SAD: 7.58，MSE: 9.49，模型为stage2.pdparams。

* **stage3：使用个人数据集微调。

注：本次比赛提交的是stage2模型，因为训练所用数据集都为公开数据集，方便复现。
原作者在论文中也使用了个人数据集微调，但没有公开。因此我增加了自己数据进行训练，没有条件的同学可以利用原工程生成pha作为训练数据。
模型最好精度为SAD: 7.61，MSE: 9.47，模型为stage3.pdparams。



```python
# [VideoMatte240K & PhotoMatte85 数据集](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
# [Distinctions646_person 数据集](https://github.com/cs-chan/Total-Text-Dataset)
# 数据集需要申请，请自行下载

! ./run.sh

```

# 验证




```python
# 解压测试集
!unzip ./data/data111962/PhotoMatte85_eval.zip -d ./data/
```


```python
!python eval.py
```

    W1013 17:35:31.830500   406 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1013 17:35:31.835165   406 device_context.cc:422] device: 0, cuDNN Version: 7.6.
      0%|                                                    | 0/85 [00:00<?, ?it/s]/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:239: UserWarning: The dtype of left and right variables are not the same, left dtype is paddle.float32, but right dtype is paddle.bool, the right dtype will convert to paddle.float32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:239: UserWarning: The dtype of left and right variables are not the same, left dtype is paddle.float32, but right dtype is paddle.float64, the right dtype will convert to paddle.float32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    100%|███████████████████████████████████████████| 85/85 [00:28<00:00,  2.96it/s]
    paddle output:  SAD: 8.519970015918508, MSE: 9.885075489212484


# 预测


```python
!python predict.py
```

    W1013 18:00:01.562386  1535 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1013 18:00:01.567060  1535 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    save results：./image/01_pred.jpg


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
