import pandas as pd
import scipy
#将特征单独形成一个数据框
te= pd.concat((cossim1,word2vecsim1,jaccardsim1,tanimotosim1,eucldist_vectorized1,cosresult1), axis=1)
te=pd.read_csv("chusaitezheng.csv")		#写出特征成csv格式文件保存备用

#	切分训练集与测试集
from sklearn import model_selection as ms
X_train,X_test,y_train,y_test = ms.train_test_split(te2,data[4],test_size=0.1,random_state=10)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
clf.fit(X_train, y_train)
test_pro = clf.predict_proba(X_test)
test_pro1=pd.DataFrame(test_pro)

from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test,test_pro1.iloc[:,1] )#验证集上的auc值
print( test_auc )
