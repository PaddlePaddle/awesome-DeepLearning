# -*- encoding:utf-8 -*-
import pandas as pd

class FeatureDictionary(object):

    '''

    这个只是用来制作特征字典的吗？

    首先，创建一个特征处理的字典。在初始化方法中，传入第一步读取得到的训练集和测试集。然后生成字典，在生成字典中，
    循环遍历特征的每一列，如果当前的特征是数值型的，直接将特征作为键值，和目前对应的索引作为 value 存到字典中。
    如果当前的特征是 categories ，统计当前的特征总共有多少个不同的取值，这时候当前特征在字典的 value
    就不是一个简单的索引了，value 也是一个字典，特征的每个取值作为 key，对应的索引作为 value，组成新的字典。
    总而言之，这里面主要是计算了特征的的维度，numerical 的特征只占一位，categories 的特征有多少个取值，就占多少位。


    好吧，one-hot上的特征区别就体现在这里了，对于离散特征 把每个取值都当做key了
    '''
    def __init__(self,trainfile=None,testfile=None,
                 dfTrain=None,dfTest=None,numeric_cols=[],
                 ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"

        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()




    def gen_feat_dict(self):
        '''
        创建特征字典，计算了特征的的维度，numerical 的特征只占一位，categories 的特征有多少个取值，就占多少位
        '''
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)

        else:
            dfTrain = self.dfTrain

        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)

        else:
            dfTest = self.dfTest

        df = pd.concat([dfTrain,dfTest])

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1

            else:
                # 这里是特别的 针对  离散型特征的处理，  这里假定对每个one-hot过的特征值都当做特征来使用
                us = df[col].unique()
                print(us)
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                tc += len(us)
        print('feat_dict:',self.feat_dict)
        self.feat_dim = tc


class DataParser(object):

    '''
    这里的作用，可以主要观察下结果，比如下面对 验证集进行生成的 特征索引和特征值结果：

    特征索引（已转 one-hot过的）
    [180, 186, 200, 202, 205, 213, 215, 217, 219, 221, 223, 225, 227, 229, 235, 248, 250, 251, 253, 254, 255, 3, 14, 16, 19, 30, 40, 50, 54, 56, 61, 147, 66, 172, 173, 175, 176, 0, 174]
    [181, 185, 190, 202, 205, 213, 216, 217, 220, 221, 223, 225, 227, 229, 242, 248, 250, 251, 253, 254, 255, 7, 14, 16, 19, 31, 33, 50, 54, 55, 61, 79, 66, 172, 173, 175, 176, 0, 174]

    对应索引位置特征值记录： （可以发现要么是对应 为one-hot标识1，要么对应离散值。  说明那种ont-hot为0的情况是不记录的，不像通常lgb思路中，都做记录）
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.3, 0.6103277807999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.316227766, 0.6695564092, 0.3521363372, 3.4641016150999997, 2.0, 0.4086488773474527],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 0.7713624309999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.316227766, 0.6063200202000001, 0.3583294573, 2.8284271247, 1.0, 0.4676924847454411]
    '''
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict

    def parse(self,infile=None,df=None,has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"


        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id','target'],axis=1,inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'],axis=1,inplace=True)
        # dfi for feature index    特征索引
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)  对应特征值
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col,axis=1,inplace=True)
                dfv.drop(col,axis=1,inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()

        if has_label:
            return xi,xv,y
        else:
            return xi,xv,ids


