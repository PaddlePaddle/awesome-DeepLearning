
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist
from sklearn import model_selection as ms
#导入数据
data=pd.read_csv("train_data.csv",header=-1)
y = data.iloc[:,4].values

#提前预备好数据框，作为存储特征的容器
cossim1= pd.DataFrame()
ChebyDistance1=pd.DataFrame()
ManhatDistance1=pd.DataFrame()
CorDistance1=pd.DataFrame()
eucldist_vectorized1=pd.DataFrame()
	
#利用词频逆文档矩阵提取特征
for i in range(len(data)):
    global datacos1
    global datacos2									#变量声明
    datacos1 = pd.Series(data.iloc[i,1])	#提取同一行的两列（query列和title列）
    datacos2 = pd.Series(data.iloc[i,3])
    d = pd.concat([datacos1,datacos2],ignore_index=True) #将两列合并为一列
    vec = TfidfVectorizer()
    Y = vec.fit_transform(d.values)
    m = Y.todense()									#m是query和title的词频逆文档矩阵
    Y1 = m[:1]											#Y1是query列的词频逆文档特征
    Y2 = m[1:]											#Y2是title列的词频逆文档特征
 	 #计算余弦相似度
    X1 = np.vstack([Y1,Y2])
    cossim11 = pd.DataFrame(1 - pdist(X1,'cosine'))
    cossim1=cossim1.append(cossim11)	#将每一行提取的余弦相似度累加在表格下一行存储起来
 	 #计算切比雪夫距离、曼哈顿距离、欧几里得距离、皮尔逊相关系数（按照出现次序排列）
    Y3=np.array(Y1)
    Y4=np.array(Y2)
    ChebyDistance11=pd.DataFrame(pd.Series(np.max(np.abs( Y3-Y4))))
    ChebyDistance1=ChebyDistance1.append(ChebyDistance11)
   
    ManhatDistance11=pd.DataFrame(pd.Series(np.sum(np.abs(Y3-Y4))))
    ManhatDistance1=ManhatDistance1.append( ManhatDistance11)
   
    eucldist_vectorized11= np.sqrt(np.sum((Y3 - Y4)**2))
    eucldist_vectorized11= pd.DataFrame(pd.Series(eucldist_vectorized11))
    eucldist_vectorized1=eucldist_vectorized1.append(eucldist_vectorized11) 
  
    CorDistance11=pd.DataFrame(pd.Series(1-np.corrcoef(Y3,Y4)[0,1]))
    CorDistance1=CorDistance1.append(CorDistance11)

#注意，这里已经出for循环了，我们将得到的特征进行填充缺失值处理。
#缺失值出现的原因请读者自己探索。
cossim1=cossim1.reset_index(drop=True)
cossim1=cossim1.fillna(0)
eucldist_vectorized1=eucldist_vectorized1.reset_index(drop=True)
ChebyDistance1= ChebyDistance1.reset_index(drop=True)
ManhatDistance1=ManhatDistance1.reset_index(drop=True)
CorDistance1=CorDistance1.reset_index(drop=True)
CorDistance1=CorDistance1.fillna(0)

