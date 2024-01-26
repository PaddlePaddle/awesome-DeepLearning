### 数据集介绍

如今近视已经成为困扰人们健康的一项全球性负担，在近视人群中，有超过35%的人患有重度近视。近视会拉长眼睛的光轴，也可能引起视网膜或者络网膜的病变。随着近视度数的不断加深，高度近视有可能引发病理性病变，这将会导致以下几种症状：视网膜或者络网膜发生退化、视盘区域萎缩、漆裂样纹损害、Fuchs斑等。因此，及早发现近视患者眼睛的病变并采取治疗，显得非常重要。

`iChallenge-PM`是百度大脑和中山大学中山眼科中心联合举办的`iChallenge`比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。  
其中训练集名称第一个字符表示类别，如下图9 所示。  
![图9 train data](https://ai-studio-static-online.cdn.bcebos.com/e6c61f9425d14269a9e24525aba5d32a363d16ed74834d11bf58f4be681814f2)  
图9 train data  

H：高度近视HighMyopia  
N：正常视力Normal  
P：病理性近视Pathologic Myopia  

**P是病理性近似，正样本，类别为1；H和N不是病理性近似，负样本，类别为0。**

验证集的类别信息储存在PALM-Validation-GT的PM_Label_and_Fovea_Location.xlsx文件中，如下图9 所示。  
![图10 validation](https://ai-studio-static-online.cdn.bcebos.com/53a6f31c7d5a4de0a7927bc66901a4d23b1b69bcd39543e99bf42ca11a2203bc)  
图10 validation  

其中`imgName`列表示图片的名称，`Label`列表示图片对应的标签。

### 数据解压

源数据链接：https://aistudio.baidu.com/aistudio/datasetdetail/19469

在`aistudio`平台通过以下代码进行数据集的解压。

```
if not os.path.isdir("train_data"):
    os.mkdir("train_data")
else:
    print('Train_data exist')
if not os.path.isdir('PALM-Training400'):
    !unzip -oq /home/aistudio/data/data19469/training.zip
    !unzip -oq /home/aistudio/data/data19469/validation.zip
    !unzip -oq /home/aistudio/data/data19469/valid_gt.zip
    !unzip -oq /home/aistudio/PALM-Training400/PALM-Training400.zip -d /home/aistudio/train_data/
else:
    print('The data has been decompressed')
```

解压后的文件目录级别为：

1./home/aistudio/trian_data

2./home/aistudio/PALM-Training400

3./home/aistudio/PALM-Validation400

4./home/aistudio/PALM-Validation-GT

训练数据位置：/home/aistudio/train_data/PALM-Training400/

验证集位置：/home/aistudio/PALM-Validation400

验证集标签位置：/home/aistudio/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx

可以通过以下代码查看训练集情况：

```
# 查看训练集
! dir /home/aistudio/train_data/PALM-Training400/
```

