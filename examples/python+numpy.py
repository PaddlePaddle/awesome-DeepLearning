#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
data = pd.read_csv('train.csv')
# 查看数据
data.head()
# 查看数据集形状
data.shape
# 查看数据集数据类型
data.dtypes


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
# 设置图幅大小
pylab.rcParams['figure.figsize'] = (15, 10)
# 计算相关系数
corrmatrix = data.corr()
# 绘制热力图，热力图横纵坐标分别是data的index/column,vmax/vmin设置热力图颜色标识上下限，center显示颜色标识中心位置，cmap颜色标识颜色设置
sns.heatmap(corrmatrix,square=True,vmax=1,vmin=-1,center=0.0,cmap='coolwarm')


# In[ ]:


# 取相关性前10的特征
k=10
# data.nlargest(k, 'target')在data中取‘target'列值排前十的行
# cols为排前十的行的index,在本例中即为与’SalePrice‘相关性最大的前十个特征名
cols = corrmatrix.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
#data[cols].values.T
#设置坐标轴字体大小
sns.set(font_scale=1.25)
# sns.heatmap() cbar是否显示颜色条，默认是；cmap显示颜色；annot是否显示每个值，默认不显示；
# square是否正方形方框，默认为False,fmt当显示annotate时annot的格式；annot_kws为annot设置格式
# yticklabels为Y轴刻度标签值，xticklabels为X轴刻度标签值
hm = sns.heatmap(cm,cmap='RdPu',annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
 
 
# 上例提供了求相关系数另一种方法，也可以直接用data.corr(),更方便
cm1 = data[cols].corr()
hm2 = sns.heatmap(cm1,square=True,annot=True,cmap='RdPu',fmt='.2f',annot_kws={'size':10})


# In[ ]:


cols1 = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(data[cols1],size=2.5)


# In[ ]:


# isnull() boolean, isnull().sum()统计所有缺失值的个数
# isnull().count()统计所有项个数（包括缺失值和非缺失值），.count()统计所有非缺失值个数
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
# pd.concat() axis=0 index,axis=1 column, keys列名
missing_data = pd.concat([total,percent],axis=1,keys = ['Total','Percent'])
missing_data.head(20)


# In[ ]:


# 处理缺失值，将含缺失值的整列剔除
data1 = data.drop(missing_data[missing_data['Total']>1].index,axis=1)
# 由于特征Electrical只有一个缺失值，故只需删除该行即可
data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
# 检查缺失值数量
data2.isnull().sum().max()


# In[ ]:


feature_data = data2.drop(['SalePrice'],axis=1)
target_data = data2['SalePrice']
 
# 将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)


# In[ ]:


from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
 
df_train = pd.concat([X_train,y_train],axis=1)
# ols("target~feature+C(feature)", data=data
# C(feature)表示这个特征为分类特征category
lr_model = ols("SalePrice~C(OverallQual)+GrLivArea+C(GarageCars)+TotalBsmtSF+C(FullBath)+YearBuilt",data=df_train).fit()
print(lr_model.summary())
 
# 预测测试集
lr_model.predict(X_test)


# In[ ]:


# prstd为标准方差，iv_l为置信区间下限，iv_u为置信区间上限
prstd, iv_l, iv_u = wls_prediction_std(lr_model, alpha = 0.05)
# lr_model.predict()为训练集的预测值
predict_low_upper = pd.DataFrame([lr_model.predict(),iv_l, iv_u],index=['PredictSalePrice','iv_l','iv_u']).T
predict_low_upper.plot(kind='hist',alpha=0.4)


# In[ ]:





# In[ ]:




