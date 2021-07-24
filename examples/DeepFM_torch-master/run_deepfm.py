# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from collections import OrderedDict, namedtuple, defaultdict
import random
from deepctrmodels.deepfm import Deepfm






if __name__ == "__main__":

    seed = 1024
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]   #C代表类别特征 class
    dense_features =  ['I' + str(i) for i in range(1, 14)]   #I代表数值特征 int
    col_names = ['label'] + dense_features + sparse_features
    data = pd.read_csv('criteo_sampled_data.csv', names=col_names, sep='\t')
    # data = pd.read_csv('criteo_train_1m.txt', names=col_names, sep='\t')
    # data = pd.read_csv('total.txt')
    feature_names = sparse_features + dense_features         #全体特征名
    data[sparse_features] = data[sparse_features].fillna('-1', )   # 类别特征缺失 ，使用-1代替
    data[dense_features] = data[dense_features].fillna(0, )        # 数值特征缺失，使用0代替
    target = ['label']                                             # label

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # 使用LabelEncoder()，为类别特征的每一个item编号
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 数值特征 max-min 0-1归化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    feat_sizes1={ feat:1 for feat in dense_features}
    feat_sizes2 = {feat: len(data[feat].unique()) for feat in sparse_features}
    feat_sizes={}
    feat_sizes.update(feat_sizes1)
    feat_sizes.update(feat_sizes2)
    # print(feat_sizes)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2,random_state=2020)
    # print(train.head(5))
    # print(test.head(5))
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input =  {name: test[name]  for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = Deepfm(feat_sizes ,sparse_feature_columns = sparse_features,dense_feature_columns = dense_features,
                   dnn_hidden_units=[400,400,400] , dnn_dropout=0.9 , ebedding_size = 8 ,
                   l2_reg_linear=1e-3, device=device)



    model.fit(train_model_input, train[target].values , test_model_input , test[target].values ,batch_size=50000, epochs=150, verbose=1)

    pred_ans = model.predict(test_model_input, 50000)

    print("final test")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))