# -*- encoding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt

import config
from metrics import gini_norm
from DeepFM import DeepFM
'''
      优点：
            （1） 不需要预训练 FM 得到隐向量；
            （2） 不需要人工特征工程；
            （3）能同时学习低阶和高阶的组合特征；
            （4）FM 模块和 Deep 模块共享 Feature Embedding 部分，可以更快的训练，以及更精确的训练学习。
  
      感觉这里唯一的难点就是在   DeepFM中对网络结构的理解，其他都不是问题，
              deep+FM两部分， 相同的特征embeding嵌入，不考虑域的概念，分别进行低阶特征抽取、高阶特征交互。
         
      感觉这才是一个  比较好的项目源码，可调控性非常强，既可以做FM 也可以做  deep模型、deepFM模型。 
      
      
      好吧，没难度了，就这几个过程：
           （1）数据的加载。 训练集和测试集的预处理、分割
           （2）对模型的构建，其中deepFM的构建，包括参数初始化、前置特征embedidng处理、FM一阶处理、FM二阶处理、Deep高阶nn处理  
           三部分输出相加并sigmoid的过程。
            (3) 模型的损失计算及优化。
            
    
     这里有个点embeding层到底是怎么做的呢。
           feature_size: 256     经过onehot后的特征数量，会对这个特征维度做embeding转换
           
           在程序中主要是在声明变量时候加以使用
           field_size: 39        这里的field_size是什么意思，在deepFM也有域的概念？这里表示在进行特征one-hot之前特征的数量，
                                  好吧，这可能就是field最本质的含义，用来表示进行one-hot之前的特征情况， 那么这样含义可能就是
                                  身高和性别不是同一field,而 性别=男与 性别=女 属于同一field 。
           被大量使用，特征首先是被转成这个形式的。
           
           
           在FM部分的输入， 一次项对应输入长度是 39， 二次项对应输入长度是256 。
           
           
           emmm....这个源码还是相当复杂的，要琢磨的点好多啊，首先Xv_train_  就不知道怎么获取到的， 还有field_size 是怎么使用的，为什么
           需要这一含义......等等，好多问题啊、   为什么分field_size和feature_size使用，FeatureDictionary怎么包装和解析的都是问题啊
                                  
        
           慢慢看吧，也算有点了解了，这里确实用到了不同域field 进行分别处理的思想。
           
           
           ok,都搞明白了，只要理解好field_size  和  embedding-size  以及   feature_size 的含义和区别  就能懂了。
      

'''
def load_data():
    '''
        加载数据，这个也是非常重要的。

        会发现这里好像有缺失值的统计， config.IGNORE_COLS  和config.CATEGORICAL_COLS的筛选


        利用 pandas 读取数据，然后获得对应的特征和 target ，保存到对应的变量中。并且将 categories 的变量保存下来。
    '''
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ['id','target']]
        #df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ['id','target']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['id'].values

    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices

def run_base_model_dfm(dfTrain,dfTest,folds,dfm_params):
    '''
     对模型的运行部分，  可以往下看发现，这部分  同时可以用于   设置使用FM 、Deep 、DeepFM这三种不同的模型
    '''

    #  别忽视了  FeatureDictionary 这里面有非常多的信息包装 转换的。 这里 解析 和字典包装真的是有点不明白，太复杂了，v是怎么获取使用的
    fd = FeatureDictionary(dfTrain=dfTrain,
                           dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols = config.IGNORE_COLS)
    # 在解析数据中，逐行处理每一条数据，dfi 记录了当前的特征在总的输入的特征中的索引。dfv 中记录的是具体的值，
    # 如果是 numerical 特征，存的是原始的值，如果是 categories 类型的，就存放 1。这个相当于进行了 one-hot 编码，
    # 在 dfi 存储了特征所在的索引。输入到网络中的特征的长度是      ( numerical 特征的个数 +categories 特征 one-hot 编码的长度 )。
    # 最终，Xi 和 Xv 是一个二维的 list，里面的每一个 list 是一行数据，Xi 存放的是特征所在的索引，Xv 存放的是具体的特征值。
    data_parser = DataParser(feat_dict= fd)

    # Xi_train ：列的序号
    # Xv_train ：列的对应的值

    # 解析数据 Xi_train 存放的是特征对应的索引 Xv_train 存放的是特征的具体的值
    Xi_train,Xv_train,y_train = data_parser.parse(df=dfTrain,has_label=True)
    Xi_test,Xv_test,ids_test = data_parser.parse(df=dfTest)


    #这里面是二维的，  大列表是 每个样本，小列表表示具体对应feature_index下的value的长度  。 小列表长度应该不是统一的，因为针对one-hot，只显示为1的
    print('Xi_train:',Xi_train)    #存储了对应标签索引
    print('Xv_train:', Xv_train)   #存储了真实值
    print('y_train:', y_train)
    print('Xi_test:', Xi_test)
    print('Xv_test:', Xv_test)

    print('Xi_train shape:', len(Xi_train))  # 存储了对应标签索引
    print('Xv_train shape:', len(Xv_train))  # 存储了真实值
    print('y_train shape:', len(y_train))
    print('Xi_test shape:', len(Xi_test))
    print('Xv_test shape:', len(Xv_test))
    #print('ids_test:', ids_test)
    print(dfTrain.dtypes)

    #field_size  是原始的特征size，   feature_size是经过对离散型数据one-hot处理后的特征数量
    dfm_params['feature_size'] = fd.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0],1),dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0],1),dtype=float)

    _get = lambda x,l:[x[i] for i in l]

    gini_results_cv = np.zeros(len(folds),dtype=float)
    gini_results_epoch_train = np.zeros((len(folds),dfm_params['epoch']),dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds),dfm_params['epoch']),dtype=float)

    for i, (train_idx, valid_idx) in enumerate(folds):

        #   这里Xi_train_, Xv_train_, y_train_ 分别表示当前的特征在总的输入的特征中的索引、特征的具体的值、对应的标签索引
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        # 训练好模型 并进行预测
        dfm = DeepFM(**dfm_params)

        print('before fit   Xi_train_:', Xi_train_[0:3])
        print('before fit   Xv_train_:', Xv_train_[0:3])
        print('before fit   y_train_:', y_train_[0:3])
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta

def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("fig/%s.png"%model_name)
    plt.close()




print('---------     一切由此开始     -----------')
dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,   #   对特征embedding的大小， 估计是隐向量的长度
    "dropout_fm":[1.0,1.0],
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "l2_reg":0.01,
    "verbose":True,
    "eval_metric":gini_norm,
    "random_seed":config.RANDOM_SEED
}

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

#y_train_dfm,y_test_dfm = run_base_model_dfm(dfTrain,dfTest,folds,dfm_params)
y_train_dfm, y_test_dfm = run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)


# ------------------ FM Model ------------------
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
y_train_fm, y_test_fm = run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


# ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
y_train_dnn, y_test_dnn = run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)