from math import*
from decimal import Decimal
import pandas as pd
import numpy as np
data=pd.read_csv("train_data.csv",header=-1)

def jaccard(a,b):
    a=ws3
    b=ws4
    ret1 = [ i for i in a if i not in b ]
    ret11=str(ret1)
    ret12=ret11.split(" ")
    lenret12=len(ret12)
    union1=set(a).union(set(b))
    union11=str(union1)
    union12=union11.split(" ")
    lenunion12=len(union12)
    jaccardsim=lenret12/lenunion12
    return jaccardsim

def tanimoto(p,q):
    moto1 = [v for v in p if v in q]
    moto11=str(moto1)
    moto12=moto11.split(" ")
    lenmoto12=len(moto12)
    p11=str(p)
    p12=p11.split(" ")
    lenp12=len(p12)
    q11=str(q)
    q12=q11.split(" ")
    lenq12=len(q12)
    return lenmoto12 / (lenq12 + lenp12 - lenmoto12)

ws1=np.array(data[1])
ws2=np.array(data[3])
ws1=ws1.tolist()
ws2=ws2.tolist()
jaccardsim1=pd.DataFrame()
tanimotosim1=pd.DataFrame()
for i in range(len(data)):
    ws3=[ws1[i]]
    ws4=[ws2[i]]
    jaccardsim11=jaccard(ws3,ws4)
    jaccardsim11=pd.DataFrame(pd.Series(jaccardsim11))
    jaccardsim1=jaccardsim1.append(jaccardsim11)
    tanimotosim11=tanimoto(ws3,ws4)
    tanimotosim11=pd.DataFrame(pd.Series(tanimotosim11))
    tanimotosim1=tanimotosim1.append(tanimotosim11)
jaccardsim1=jaccardsim1.reset_index(drop=True)
tanimotosim1=tanimotosim1.reset_index(drop=True)


