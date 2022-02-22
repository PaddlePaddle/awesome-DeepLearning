# 评估指标

机器学习的评价指标有精度、精确率、召回率、P-R曲线、F1 值、TPR、FPR、ROC、AUC等指标，还有在生物领域常用的敏感性、特异性等指标。

## 基础

在分类任务中，各指标的计算基础都来自于对正负样本的分类结果，用混淆矩阵表示，如 **图1** 所示：

<center><img src="https://raw.githubusercontent.com/w5688414/paddleImage/main/metrics_img/confusion_metric.png" width="500" hegiht="" ></center>
<center><br>图1 混淆矩阵 </br></center>

## 精度


$$Accuracy=\frac{TP+TN}{TP+FN+FP+TN}$$

即所有分类正确的样本占全部样本的比例。

## 精确率
精准率又叫做：Precision、查准率

$$Precision=\frac{TP}{TP+FP}$$

即预测是正例的结果中，确实是正例的比例。

## 召回率

召回率又叫：Recall、查全率

$$Recall=\frac{TP}{TP+FN}$$

即所有正例的样本中，被找出的比例

## P-R曲线

P-R曲线又叫做：PRC

<center><img src="https://raw.githubusercontent.com/w5688414/paddleImage/main/metrics_img/PRC.png" width="500" hegiht="" ></center>
<center><br>图2 PRC曲线图</br></center>

根据预测结果将预测样本排序，最有可能为正样本的在前，最不可能的在后，依次将样本预测为正样本，分别计算当前的精确率和召回率，绘制P-R曲线。

## F1 值

$$F1=\frac{2 * P * R}{P + R}$$

## TPR

真正例率，与召回率相同

$$TPR=\frac{TP}{TP+FN}$$

## FPR

假正例率

$$FPR=\frac{FP}{TN+FP}$$

## ROC

受试者工作特征

根据预测结果将预测样本排序，最有可能为正样本的在前，最不可能的在后，依次将样本预测为正样本，分别计算当前的TPR和FPR，绘制ROC曲线。

## AUC

Area Under ROC Curve，ROC曲线下的面积：

<center><img src="https://raw.githubusercontent.com/w5688414/paddleImage/main/metrics_img/AUC.png" width="500" hegiht="" ></center>
<center><br>图3 ROC曲线图</br></center>

## 敏感性

敏感性或者灵敏度（Sensitivity，也称为真阳性率）是指实际为阳性的样本中，判断为阳性的比例（例如真正有生病的人中，被医院判断为有生病者的比例），计算方式是真阳性除以真阳性+假阴性（实际为阳性，但判断为阴性）的比值（能将实际患病的病例正确地判断为患病的能力，即患者被判为阳性的概率）。公式如下：

$$sensitivity =\frac{TP}{TP + FN}$$

即有病（阳性）人群中，检测出阳性的几率。（检测出确实有病的能力）

## 特异性

特异性或特异度（Specificity，也称为真阴性率）是指实际为阴性的样本中，判断为阴性的比例（例如真正未生病的人中，被医院判断为未生病者的比例），计算方式是真阴性除以真阴性+假阳性（实际为阴性，但判断为阳性）的比值（能正确判断实际未患病的病例的能力，即试验结果为阴性的比例）。公式如下：

$$specificity =\frac{TN}{TN + FP}$$

即无病（阴性）人群中，检测出阴性的几率。（检测出确实没病的能力）
