import os
import numpy as np
class DeepFM_Dataset():
    #读取数据集中的数据
    def __init__(self,batchsize,dataFileDir,sparse_num_field=26,dense_feature_dim=13):
        self.batchsize=batchsize
        self.dataFileDir=dataFileDir
        self.dataFiles=os.listdir(dataFileDir)
        self.next_batch_reader=self.nextBatch()
    #获得下一批数据
    def nextBatch(self,sparse_num_field=26,dense_feature_dim=13):
        batchData=[]
        cnt=0
        #从所有的文件中一行一行读取
        for dataFile in self.dataFiles:
            with open(self.dataFileDir.strip('/')+'/'+dataFile,'r') as lines:
                for line in lines:
                    batchData.append(predata(line))
                    cnt+=1
                    if cnt % self.batchsize==0:
                        batchDataArray=np.array(batchData)
                        #提取label，稀疏特征，稠密特征
                        label,sparse_feature,dense_feature=\
                                        batchDataArray[:,0].astype(np.float32),\
                                        batchDataArray[:,1:sparse_num_field+1].astype(np.int),\
                                        batchDataArray[:,-dense_feature_dim:].astype(np.float32)
                        yield  label,sparse_feature,dense_feature
                        batchData=[]

    def getNextBatchData(self):
        return next(self.next_batch_reader)# [[label,sparse fea,dense fea],...]   shape:[batchsize,1+26+13]