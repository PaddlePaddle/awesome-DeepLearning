## 数据文件准备

数据集已挂载至aistudio项目中，如果需要本地训练可以从这里下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/105347)，和[标签](https://aistudio.baidu.com/aistudio/datasetdetail/103218)文件

数据集目录大致如下，可根据实际情况修改
```
Data
|-- coco
|   |-- annotions
|   |-- images
|      |-- train2017
|      |-- val2017
|      |-- test2017
|   |-- labels
|      |-- train2017
|      |-- val2017
|      |-- train2017.cache(初始解压可删除，训练时会自动生成)
|      |-- val2017.cache(初始解压可删除，训练时会自动生成)
|   |-- test-dev2017.txt
|   |-- val2017.txt
|   |-- train2017.txt
`   `-- validation
```

## 训练

### 单卡训练
```
python train.py --batch-size 8 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights '' --sync-bn --device 0 --name yolov4-p5
```
![](https://ai-studio-static-online.cdn.bcebos.com/bb2b5f39e95d4272ab90ae98ae3f46f45d5078c269074176836dc957f295ec84)

### 多卡训练
```
python train_multi_gpu.py --batch-size 12 --img 896 896 --data coco.yaml --cfg yolov4-p5.yaml --weights '' --sync-bn --name yolov4-p5 --notest
```
多卡训练项目已提交至[脚本任务ScaledYOLOv4](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2326709)

多卡训练日志以及模型可在[此处](https://pan.baidu.com/s/1JIW1FtFNymwK4gP_VNDVPA)下载，提取码：mxz8



### 验证
确保已安装`pycocotools`
```
pip install pycocotools
```
```
python test.py --img 896 --conf 0.001 --batch 8  --data coco.yaml --weights scaledyolov4.pdparams
```
需要注意到，在test.py的58行指定模型配置文件路径`model = Model('/home/aistudio/ScaledYOLOv4-yolov4-large/models/yolov4-p5.yaml', ch=3, nc=80)`以及227行的标签路径`cocoGt = COCO(glob.glob('/home/aistudio/Data/coco/annotations/instances_val2017.json')[0])`，运行后会出现`test_batch0_gt.jpg`和`test_batch0_pred.jpg`

#### 验证结果如下所示

<center>**GroundTruth**</center>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b041b1556ecf45a6aba198b2cbd041bef47adb049bfa48848169274393a2f827" width="600"/></center>

<center>**Pred**</center>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/04fcf8bc8cae4937b2bfd4bef9851437ef23f40235be419aa02ca63414ca97af" width="600"/></center>

验证完成后会生成`detections_val2017__results.json`并打印验证信息

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/6993bba5ed314f2d89763828073fea828f16b696ea954c15b587fa2e14daeb32" width="1200"/></center>

### 推理

```
python detect.py
```
与验证相同，你需要指定detec.py中33行模型配置文件路径，在inference下放置了一些测试图像，运行结果将会保存在inference/output文件夹下

![](https://ai-studio-static-online.cdn.bcebos.com/97f67f7cb9744b37ac95cc4a5566244c217b4588b6574fccb229e21435264989)

![](https://ai-studio-static-online.cdn.bcebos.com/96b2a54f471e457d9a1a294d292f03945e4cc912f41d4072bbb1453a5c96739b)
![](https://ai-studio-static-online.cdn.bcebos.com/79da74ec819647aba31f984b82aa70c1ffa48884203540ea8fdf584f21479120)

![](https://ai-studio-static-online.cdn.bcebos.com/64c46191443d42109778ec19a343a1adf3cd9e3adf4e44a3aae4896944b2a0f5)
![](https://ai-studio-static-online.cdn.bcebos.com/00d648e60896462aa25fe3c2a468715cf653b81abb8f4e718a02d999799f469f)

#### [GitHub地址](https://github.com/GuoQuanhao/ScaledYOLOv4-Paddle)

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- |
| 学校        | 电子科技大学研2020级     |
| 研究方向     | 计算机视觉             |
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
