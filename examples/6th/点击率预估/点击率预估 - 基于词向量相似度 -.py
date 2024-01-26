
import gensim
import pandas as pd
import numpy as np
data=pd.read_csv("train_data.csv",header=-1)
dataword2vec2 = pd.concat((data[1],data[3]), axis=1)
dataword2vec3=np.array(dataword2vec2)
dataword2vec3=dataword2vec3.tolist()		  #必须用列表类型的数据才能训练词向量
model = word2vec.Word2Vec(dataword2vec3, size=300, hs=1, min_count=1, window=3)
ws1=np.array(dataword2vec2[1])
ws2=np.array(dataword2vec2[3])
ws1=ws1.tolist()
ws2=ws2.tolist()																	
word2vecsim1=pd.DataFrame()
for i in range(len(data)):
    ws3=[ws1[i]]
    ws4=[ws2[i]]
    word2vecsim2=model.wv.n_similarity(ws3,ws4)		#计算两列的相似度
    word2vecsim2=pd.DataFrame(pd.Series(word2vecsim2))
    word2vecsim1=word2vecsim1.append(word2vecsim2)
word2vecsim1=word2vecsim1.reset_index(drop=True)
