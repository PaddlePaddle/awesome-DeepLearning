# 智慧商超商品识别方案

## 1. 项目介绍

   目前，在传统的商超零售企业的经营过程中，急需通过引进数字化及人工智能等新兴技术，进行管理能力、成本控制、用户体验等多维度的全面升级。而现如今普遍通用的人工智能技术并不能帮助零售企业从根本上上述问题。因此传统商超零售企业数字化转型陷入两难境地。   
<div style="align: center">
<img src="./image/background.jpg">
</div>

**痛点问题**

   1. **结算效率要求极高**：在商超零售场景中，若顾客购买的商品较多，采用传统的条形码结算，效率较低，顾客购物体验较差；
   2. **不同商品相似度极高**：比如同一种饮料的不同口味，就很可能拥有非常类似的包装。而且即便对于同一件商品，**在不同情况下所获得的商品图像都往往存在相当大的差异**；
   3. **品类更新极快**：商超零售场景下，新品通常以**小时级别**速度更新迭代，每增加新产品时若仅靠单一模型均需重新训练模型，模型训练成本及时间成本极大。

**解决方案**

   PaddleClas团队开源的[图像识别PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)技术方案，主要由主体检测、特征学习和向量检索三个模块组成，是一个实用的轻量级通用图像识别系统。基于此技术方案，商超零售企业可实现大量商品的一键式智能化识别，大大提高识别效率，节省人工及时间成本。  
     
   此外，当新品迭代更新时，PP-shitu无需重新训练模型，能够做到“即增即用”，完美解决上述痛点问题，大大提高了人工智能在商超零售行业的应用落地可能性。   
   
   PP-shitu技术方案可具体应用于例如：商品结算、库存管理等关于商品识别的商超细分场景。  
   
**模型工具简介**

   飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。  

   而[PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)是一个实用的轻量级通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化8个方面，采用多种策略，对各个模块的模型进行优化，最终得到在CPU上仅0.2s即可完成10w+库的图像识别的系统。



## 2. 安装说明

**环境要求**

- Python >= 3.6
- PaddlePaddle >= 2.1
- Linux 环境最佳

- 安装PaddleClas


```python
# 克隆 PaddleClas

# github仓库
!git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.3 
# gitee仓库(推荐)
!git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.3

```

```python
# 安装 Python 依赖库  
# PaddleClas 的 Python 依赖库在 `requirements.txt` 中给出，可通过如下命令安装：
%cd /home/aistudio/PaddleClas/
!pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple 
```

## 3. 数据准备  

**数据集介绍**  

1. 目前开源的商品识别方向的数据集
- [Products-10K Large Scale Product Recognition Dataset](https://www.kaggle.com/c/products-10k/data?select=train.csv) :数据集中的所有图片均来自京东商城。数据集中共包含 1 万个经常购买的 SKU。所有 SKU 组织成一个层次结构。总共有近 19 万张图片。在实际应用场景中，图像量的分布是不均衡的。所有图像都由生产专家团队手工检查/标记。
- [RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification](https://arxiv.org/abs/2006.12634) :收集了超过 500,000 张货架上零售产品的图像，属于 2000 种不同的产品。所有图片均在实体零售店人工拍摄，自然采光，符合实际应用场景。
2. 本项目**以实际应用场景为依托，以数据质量为主要衡量标准**，主体基于上述两大开源商品识别方向数据集，从中以**样本均衡**、**图像质量**、**实际应用**等为思考维度选取了部分SKU，并结合图片爬虫技术等数据搜索方式，开源了一份更符合本项目实际应用背景和技术需求的[demo数据集](https://aistudio.baidu.com/aistudio/datasetdetail/108651)。此数据集总计覆盖商品**357类**，涵盖包括厨房用品、日用品、饮料等**生活日常购买商品**，商品类别**细粒度较高**，涉及诸如**同一品牌的不同规格商品**、**同一品类的不同品牌商品**等实际场景下的数据可能性，能够模拟实际购物场景下的购买需求。  

**商品部分**  

> 	东古酱油一品鲜  
	东古黄豆酱750G  
	东鹏特饮罐装  
	中华（硬）  
	中华（软）  
	乳酸菌600亿_2  
	乳酸菌600亿_3  
	乳酸菌600亿原味  
	乳酸菌600亿芒果  
	乳酸菌600亿芦荟  
   ...
   
**数据集格式**  

* 训练集合（train dataset）：用来训练模型，使模型能够学习该集合的图像特征。
* 底库数据集合（gallery dataset）：用来提供图像检索任务中的底库数据，该集合可与训练集或测试集相同，也可以不同，当与训练集相同时，测试集的类别体系应与训练集的类别体系相同。
* 测试数据集合（query dataset）：用来测试模型的好坏，通常要对测试集的每一张测试图片进行特征提取，之后和底库数据的特征进行距离匹配，得到识别结果，后根据识别结果计算整个测试集的指标。
训练集、底库数据集和测试数据集均使用 `txt` 文件指定，训练数据集 `train_list.txt`文件内容格式如下所示：
```shell
# 采用"空格"作为分隔符号
...
train/10/1283.jpg 10 624
train/10/1284.jpg 10 625
train/10/1285.jpg 10 626
train/10/1286.jpg 10 627
...
```
验证数据集(本数据集中既是 gallery dataset，也是 query dataset)test_list.txt 文件内容格式如下所示：
```shell
...
test/103/743.jpg 103 743
test/103/744.jpg 103 744
test/103/745.jpg 103 745
test/103/746.jpg 103 746
...
```
**注：**
1. 每行数据使用“空格”分割，三列数据的含义分别是训练数据的路径、训练数据的label信息、训练数据的unique id;
2. 本数据集中由于 gallery dataset 和 query dataset 相同，为了去掉检索得到的第一个数据（检索图片本身无须评估），每个数据需要对应一个 unique id（每张图片的 id 不同即可，可以用行号来表示 unique id），用于后续评测 mAP、recall@1 等指标。yaml 配置文件的数据集选用 VeriWild。  
根据以上描述对数据集进行处理，并将数据集修改为如下目录格式：
```
├── classlabel.txt	# 数据label和对应商品名称
├── gallery_label.txt	# 图片地址及对应的商品名称
├── gallery		# 底库图片
├── test		# 测试集图片
├── test_list.txt	# 测试集图片地址及对应的商品名称
├── train		# 训练集图片
└── train_list.txt	# 训练集图片地址及对应的商品名称
```



## 4. 模型选择
PP-ShiTu是一个实用的轻量级通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化8个方面，采用多种策略，对各个模块的模型进行优化，最终得到在CPU上仅0.2s即可完成10w+库的图像识别的系统。  

**主体检测**  

主体检测技术是目前应用非常广泛的一种检测技术，它指的是检测出图片中一个或者多个主体的坐标位置，然后将图像中的对应区域裁剪下来，进行识别，从而完成整个识别过程。主体检测是识别任务的前序步骤，可以有效提升识别精度。  
考虑到商品识别实际应用场景中，需要快速准确地获得识别结果，故本项目选取适用于 CPU 或者移动端场景的**轻量级主体检测模型**[PicoDet](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_pretrained.pdparams)作为本项目主体检测部分的模型。此模型融合了ATSS、Generalized Focal Loss、余弦学习率策略、Cycle-EMA、轻量级检测 head等一系列优化算法，基于COCO train2017数据集进行大规模预训练，最终inference模型大小(MB)仅**30.1MB**，mAP可达**40.1%**，在**cpu**下单张图片预测耗时仅**29.8ms**，完美符合本项目实际落地需求，故在本项目中不对主体检测部分做适应性训练。  

**特征提取**  

特征提取是图像识别中的关键一环，它的作用是将输入的图片转化为固定维度的特征向量，用于后续的向量检索。好的特征需要具备相似度保持性，即在特征空间中，相似度高的图片对其特征相似度要比较高（距离比较近），相似度低的图片对，其特征相似度要比较小（距离比较远）。Deep Metric Learning用以研究如何通过深度学习的方法获得具有强表征能力的特征。  
考虑到本项目的真实落地的场景中,推理速度及预测准确率是考量模型好坏的重要指标，所以本项目采用 [PP_LCNet_x2_5](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models/PP-LCNet.md) 作为骨干网络， Neck 部分选用 Linear Layer, Head 部分选用 ArcMargin，Loss 部分选用 CELoss，并结合度量学习**arcmargin**算法，对高相似物体的区分效果远超单一模型，能更好地适应 Intel CPU，不仅准确率超越大模型ResNet50，预测速度还能快3倍。  

**向量检索** 

向量检索技术在图像识别、图像检索中应用比较广泛。其主要目标是，对于给定的查询向量，在已经建立好的向量库中，与库中所有的待查询向量，进行特征向量的相似度或距离计算，得到相似度排序。在图像识别系统中，本项目使用 [Faiss](https://github.com/facebookresearch/faiss) 对此部分进行支持。在此过程中，本项目选取 **HNSW32** 为检索算法，使得检索精度、检索速度能够取得较好的平衡，更为贴切本项目实际应用场景的使用需求。


## 5. 模型训练
本项目在主体检测部分直接采用PaddleDetection提供给得预训练模型，故这里主要介绍特征提取部分的模型训练。  

**修改配置文件**

首先，在启动模型训练之前，需要在配置文件中修改数据配置相关的内容, 主要包括数据集的地址以及类别数量。对应到配置文件中的位置如下所示：
```
  Head:
    name: ArcMargin
    embedding_size: 512
    class_num: 358    #此处表示类别数
```
```
  Train:
    dataset:
      name: ImageNetDataset
      image_root: /home/aistudio/dataset/ #此处表示train数据所在的目录
      cls_label_path: /home/aistudio/dataset/train_list.txt  #此处表示train数据集label文件的地址
```
```
   Query:
     dataset:
       name: VeriWild
       image_root: /home/aistudio/dataset/	#此处表示query数据集所在的目录
       cls_label_path: /home/aistudio/dataset/test_list.txt #此处表示query数据集label文件的地址
```
```
   Gallery:
     dataset:
       name: VeriWild
       image_root: /home/aistudio/dataset/	#此处表示gallery数据集所在的目录
       cls_label_path: /home/aistudio/dataset/test_list.txt   #此处表示gallery数据集label文件的地址
```
**模型训练**  
- 单机单卡训练



```python
%cd /home/aistudio/PaddleClas
!python tools/train.py \
    -c ./ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu \
```

## 6.模型评估

- 单卡评估


```python
%cd /home/aistudio/PaddleClas
!python tools/eval.py \
    -c ./ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5.yaml \
    -o Global.pretrained_model="/home/aistudio/PaddleClas/output/RecModel/best_model"
```


由评估log可见，recall1为0.98232，能够符合实际产业场景应用需求。

## 7. 模型推理

    推理过程包括两个步骤： 1)导出推理模型, 2)获取特征向量

**导出推理模型**

PaddlePaddle框架保存的权重文件分为两种：支持前向推理和反向梯度的**训练模型** 和 只支持前向推理的**推理模型**。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。因此可执行如下命令导出推理模型。


```python
%cd /home/aistudio/PaddleClas
!python tools/export_model.py \
    -c ./ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5.yaml \
    -o Global.pretrained_model="output/RecModel/best_model"
```



生成的推理模型位于 inference 目录，里面包含三个文件，分别为 inference.pdmodel、inference.pdiparams、inference.pdiparams.info。 其中: inference.pdmodel 用来存储推理模型的结构, inference.pdiparams 和 inference.pdiparams.info 用来存储推理模型相关的参数信息。

**获取特征向量**


```python
%cd /home/aistudio/PaddleClas/deploy
!python python/predict_rec.py \
    -c configs/inference_rec.yaml  \
    -o Global.rec_inference_model_dir="../inference" \
    -o Global.infer_imgs="/home/aistudio/dataset/test_image/sijibao.jpg"
```


## 8.图像识别系统

这里串联主体检测、特征提取、向量检索，从而构成一整套图像识别系统：

1. 若商品为原索引库里已有的商品：

- 建立索引库  
修改`configs/build_general.yaml`文件内容：
```
Global:
  rec_inference_model_dir: "/home/aistudio/PaddleClas/inference"
```
```
IndexProcess:
  index_method: "HNSW32" # supported: HNSW32, IVF, Flat
  image_root: "/home/aistudio/dataset/"
  index_dir: "/home/aistudio/dataset/index"
  data_file:  "/home/aistudio/dataset/gallery_label.txt"
```
执行如下代码：


```python
# 建立索引库
%cd /home/aistudio/PaddleClas/deploy
!python3 python/build_gallery.py \
    -c configs/build_general.yaml \
    -o IndexProcess.data_file="/home/aistudio/dataset/gallery_label.txt" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index"
```


- 识别图片  
运行如下命令，下载通用检测`inference`模型并解压：


```python
%cd /home/aistudio/PaddleClas/deploy/
%mkdir models
%cd models
# 下载通用检测 inference 模型并解压
!wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar && tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
```


修改推理文件`configs/inference_general.yaml`内容：
```
Global:
  infer_imgs: "/home/aistudio/dataset/test_image/sijibao.jpg"
  det_inference_model_dir: "./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer"
  rec_inference_model_dir: "/home/aistudio/PaddleClas/inference"
  rec_nms_thresold: 0.05
```
```
IndexProcess:
  index_dir: "/home/aistudio/dataset/index"
  return_k: 5
  score_thres: 0.5
```

运行下面的命令，对图像 `/home/aistudio/dataset/test_image/sijibao.jpg` 进行识别与检索:


```python
#基于索引库的图像识别
%cd /home/aistudio/PaddleClas/deploy
!python python/predict_system.py \
    -c configs/inference_general.yaml \
    -o Global.infer_imgs="/home/aistudio/dataset/test_image/sijibao.jpg" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index"
```



其中 bbox 表示检测出的主体所在位置，rec_docs 表示索引库中与检测框最为相似的类别，rec_scores 表示对应的置信度。  

检测的可视化结果也保存在 output 文件夹下，对于本张图像，识别结果可视化如下所示：


<div align="center">
<img src="./image/sijibao.jpg"  width = "600" />
</div>

1. 若商品为原索引库里没有的商品：  
对图像 `/home/aistudio/dataset/test_image/recognition_2.jpg` 进行识别，
待检索图像如下所示。
<div align="center">
<img src="./image/recognition_null_test.jpg"  width = "600" />
</div>

运行如下识别命令:


```python
%cd /home/aistudio/PaddleClas/deploy
!python python/predict_system.py \
    -c configs/inference_general.yaml \
    -o Global.infer_imgs="/home/aistudio/dataset/test_image/recognition_2.jpg"
```


输出结果为**空**。  

由于默认的索引库中不包含对应的索引信息，所以这里的识别结果有误，此时我们可以通过构建新的索引库的方式，完成未知类别的图像识别。

当索引库中的图像无法覆盖我们实际识别的场景时，即在预测未知类别的图像时，只需要将对应类别的相似图像添加到索引库中，从而完成对未知类别的图像识别，这一过程是不需要重新训练的。

- 准备新的数据与标签  

   首先需要将与待检索图像相似的图像列表拷贝到索引库原始图像的文件夹。这里将所有的底库图像数据都放在文件夹 /home/aistudio/dataset/gallery/ 中。

   然后需要编辑记录了图像路径和标签信息的文本文件，这里 PaddleClas 将更正后的标签信息文件放在了 /home/aistudio/dataset/gallery_update.txt 文件中。可以与原来的 /home/aistudio/dataset/gallery_label.txt 标签文件进行对比，添加了小度充电宝和韩国进口火山泥的索引图像。

   每一行的文本中，第一个字段表示图像的相对路径，第二个字段表示图像对应的标签信息，中间用 \t 键分隔开

- 建立新的索引库
使用下面的命令构建 index 索引，加速识别后的检索过程。


```python
%cd /home/aistudio/PaddleClas/deploy/
!python python/build_gallery.py \
    -c configs/build_general.yaml \
    -o IndexProcess.data_file="/home/aistudio/dataset/gallery_update.txt" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_update"
```



最终新的索引信息保存在文件夹 `/home/aistudio/dataset/index_update` 中。

- 基于新的索引库的图像识别

使用新的索引库，对上述图像进行识别，运行命令如下:


```python
%cd /home/aistudio/PaddleClas/deploy/
!python python/predict_system.py \
    -c configs/inference_general.yaml \
    -o Global.infer_imgs="/home/aistudio/dataset/test_image/recognition_2.jpg" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_update"
```



由测试效果图可知，模型对于未参与训练的商品及多个商品均有较好的识别效果：
<!-- ![](../image/recognition_2.jpg) -->
<div align="center">
<img src="./image/recognition_2.jpg"  width = "600" />
</div>


## 9. 模型优化思路

- 检测模型调优

`PP-ShiTu`中检测模型采用的 ` PicoDet `算法，在使用官方模型后，如果不满足精度需求，则可以参考此部分文档，进行模型调优

对模型进行训练的话，需要自行准备数据，并对数据进行标注，建议一个类别至少准备200张标注图像，并将标注图像及groudtruth文件转成coco文件格式，以方便使用PaddleDetection进行训练的时候，请加载主体检测的预训练权重。


- 识别模型调优

在使用官方模型后，如果不满足精度需求，则可以参考此部分文档，进行模型调优

因为要对模型进行训练，所以参照[数据准备](#数据准备)部分描述收集自己的数据集。值得注意的是，此部分需要准备大量的数据，以保证识别模型效果。

  - 数据增强：根据实际情况选择不同数据增强方法。如：实际应用中数据遮挡比较严重，建议添加`RandomErasing`增强方法。
  - 换不同的`backbone`，一般来说，越大的模型，特征提取能力更强。
  - 选择不同的`Metric Learning`方法。不同的`Metric Learning`方法，对不同的数据集效果可能不太一样，建议尝试其他`Loss`
  - 采用蒸馏方法，对小模型进行模型能力提升
  - 增补数据集。针对错误样本，添加badcase数据

模型训练完成后，参照[测试代码](#测试代码)进行检索库更新。同时，对整个pipeline进行测试，如果精度不达预期，则重复此步骤。

## 10. 模型服务化部署

使用 PaddleServing 做服务化部署时，需要将保存的 inference 模型转换为 Serving 模型。

**模型转换**

- 将 inference 模型转换为 Serving 模型：


```python
%cd /home/aistudio/PaddleClas/deploy/
# 安装相关包
!pip3 install paddle-serving-client==0.7.0 -i https://mirror.baidu.com/pypi/simple
!pip3 install paddle-serving-app==0.7.0 -i https://mirror.baidu.com/pypi/simple
!pip3 install faiss-cpu==1.7.1post2 -i https://mirror.baidu.com/pypi/simple
!pip3 install paddle-serving-server-gpu==0.7.0.post102 -i https://mirror.baidu.com/pypi/simple 
# 执行模型转换命令
!python3 -m paddle_serving_client.convert --dirname /home/aistudio/PaddleClas/inference/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./inference_PPLCNet_serving/  \
                                         --serving_client ./inference_PPLCNet_client
```



识别推理模型转换完成后，会在当前文件夹多出 inference_PPLCNet_serving/ 和 inference_PPLCNet_client/ 的文件夹。分别修改 general_PPLCNet_x2_5_lite_v1.0_serving/ 和 general_PPLCNet_x2_5_lite_v1.0_client/ 目录下的 serving_server_conf.prototxt 中的 alias 名字： 将 fetch_var 中的 alias_name 改为 features。 修改后的 serving_server_conf.prototxt 内容如下：
```
feed_var {
  name: "x"
  alias_name: "x"
  is_lod_tensor: false
  feed_type: 1
  shape: 3
  shape: 224
  shape: 224
}
fetch_var {
  name: "save_infer_model/scale_0.tmp_1"
  alias_name: "features"
  is_lod_tensor: false
  fetch_type: 1
  shape: 512
}
```


- 转换通用检测 inference 模型为 Serving 模型：


```python
# 转换通用检测模型
%cd /home/aistudio/PaddleClas/deploy/
!python -m paddle_serving_client.convert --dirname ./models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
                                         --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
```


检测 inference 模型转换完成后，会在当前文件夹多出 picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ 和 picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/ 的文件夹。

注意: 此处不需要修改 picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ 目录下的 serving_server_conf.prototxt 中的 alias 名字。

**服务部署和请求**

注意: 识别服务涉及到多个模型，出于性能考虑采用 PipeLine 部署方式。

- 进入到工作目录


```python
%cd ./deploy/paddleserving/recognition
```

paddleserving 目录包含启动 pipeline 服务和发送预测请求的代码，包括：
```
__init__.py
config.yml                    # 启动服务的配置文件
pipeline_http_client.py       # http方式发送pipeline预测请求的脚本
pipeline_rpc_client.py        # rpc方式发送pipeline预测请求的脚本
recognition_web_service.py    # 启动pipeline服务端的脚本
```

- 启动服务


```python
# 启动服务，运行日志保存在 log.txt
!python recognition_web_service.py &>log.txt &
```

- 发送请求


```python
!python pipeline_http_client.py
```

本项目中用户提供了基于服务器的部署Demo方案。用户可根据实际情况自行参考。  

<div align="center">
<img src="./image/pic_paddle.gif"  width = "600" />
</div>
<div align="center">
<img src="./image/wx_all.png"  width = "600" />
</div>

具体可以参考：[袋鼯麻麻——智能购物平台](https://github.com/thomas-yanxin/Smart_container) 
