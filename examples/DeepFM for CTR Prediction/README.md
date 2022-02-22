项目aistudio链接：

[基于 DeepFM 模型的点击率预估模型](https://aistudio.baidu.com/aistudio/projectdetail/2251589)

# 基于 DeepFM 模型的点击率预估模型



# 一、模型简介
CTR预估是目前推荐系统的核心技术，其目标是预估用户点击推荐内容的概率。DeepFM模型包含FM和DNN两部分，FM模型可以抽取low-order特征，DNN可以抽取high-order特征。无需Wide&Deep模型人工特征工程。由于输入仅为原始特征，而且FM和DNN共享输入向量特征，DeepFM模型训练速度很快。
## 1）DeepFM模型
为了同时利用low-order和high-order特征，DeepFM包含FM和DNN两部分，两部分共享输入特征。对于特征i，标量wi是其1阶特征的权重，该特征和其他特征的交互影响用隐向量Vi来表示。Vi输入到FM模型获得特征的2阶表示，输入到DNN模型得到high-order高阶特征。

$$
\hat{y} = sigmoid(y_{FM} + y_{DNN})
$$

DeepFM模型结构如下图所示，完成对稀疏特征的嵌入后，由FM层和DNN层共享输入向量，经前向反馈后输出。

![](https://ai-studio-static-online.cdn.bcebos.com/8654648d844b4233b3a05e918dedc9b777cf786af2ba49af9a92fc00cd050ef3)

## 2）FM
FM模型不单可以建模1阶特征，还可以通过隐向量点积的方法高效的获得2阶特征表示，即使交叉特征在数据集中非常稀疏甚至是从来没出现过。这也是FM的优势所在。

$$
y_{FM}= <w,x> + \sum_{j_1=1}^{d}\sum_{j_2=j_1+1}^{d}<V_i,V_j>x_{j_1}\cdot x_{j_2}
$$

单独的FM层结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/bda8da10940b43ada3337c03332fe06ad1cd95f7780243888050023be33fc88c)

## 3）DNN
该部分和Wide&Deep模型类似，是简单的前馈网络。在输入特征部分，由于原始特征向量多是高纬度,高度稀疏，连续和类别混合的分域特征，因此将原始的稀疏表示特征映射为稠密的特征向量。

假设子网络的输出层为：
$$
a^{(0)}=[e1,e2,e3,...en]
$$
DNN网络第l层表示为：
$$
a^{(l+1)}=\sigma{（W^{(l)}a^{(l)}+b^{(l)}）}
$$
再假设有H个隐藏层，DNN部分的预测输出可表示为：
$$
y_{DNN}= \sigma{(W^{|H|+1}\cdot a^H + b^{|H|+1})}
$$
DNN深度神经网络层结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/df8159e1d56646fe868e8a3ed71c6a46f03c716ad1d74f3fae88800231e2f6d8)


## 4）Loss及Auc计算
* 预测的结果将FM的一阶项部分，二阶项部分以及dnn部分相加，再通过激活函数sigmoid给出，为了得到每条样本分属于正负样本的概率，我们将预测结果和1-predict合并起来得到predict_2d，以便接下来计算auc。
* 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。
* 该batch的损失avg_cost是各条样本的损失之和
* 我们同时还会计算预测的auc指标。



# 二、数据格式
训练及测试数据集选用Display Advertising Challenge所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。 每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中label表示广告是否被点击，点击用1表示，未点击用0表示。<integer feature>代表数值特征（连续特征），共有13个连续特征。<categorical feature>代表分类特征（离散特征），共有26个离散特征。相邻两个特征用\t分隔，缺失特征用空格表示。

  13个数值连续特征形式如下图所示：
 ![](https://ai-studio-static-online.cdn.bcebos.com/a39de00d9dfd49959d9a41e2e5493e27ca328ae6aaf04b6f91c6619ef220000d)

  26个分类离散特征形式如下图所示：
 ![](https://ai-studio-static-online.cdn.bcebos.com/87f580fa92fd456b892954fc8f1ec069f2bcb8b13de945beaa758090da9228d1)

  

```python
# 查看数据格式
# !cd slot_train_data_full/ && ls -lha && head part-0
# !cd slot_test_data_full/ && ls -lha && head part-220
import os
import subprocess
def wc_count(file_name):
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])
def wc_count_dir(dirPath):#统计最终一共有多少行数据
    cnt=0
    fileList=os.listdir(dirPath)
    for fileName in fileList:
        cnt+=wc_count(dirPath.strip('/')+'/'+fileName)
    return cnt
```



# 三、数据加载
需要对原始数据集进行处理，以作为模型的训练输入



```python
def predata(rawLine):
    '''
    数据处理，缺失值填充，原始数据拆分
    '''
    #划分特征
    padding='0'
    fea_vals=rawLine.strip().split(' ')
    label=['click']
    sparse_fea=label+[str(x) for x in range(1,27)] #提取所有的特征
    dense_fea=['dense_inputs']
    dense_fea_dim=13 #稠密特征维度为13
    slots=1+26+13
    slots_fea=sparse_fea+dense_fea
    output={}
    for fea_val in fea_vals:
        fea_val=fea_val.split(':')#根据数据格式，按":"划分
        if len(fea_val)==2:fea,val=fea_val
        else:continue
        if fea not in output.keys():output[fea]=[val]#连续特征缺失，添加新特征
        else:output[fea].append(val)#末尾添加
    
    #填充
    if len(output.keys()) != slots:
        for fea in slots_fea:
            if fea in sparse_fea:#稀疏特征
                if fea not in output.keys():output[fea]=[padding]
            elif fea not in output.keys():#连续特征完全缺失
                output[fea]=[padding]*dense_fea_dim#连续特征部分缺失
            elif len(output[fea])<dense_fea_dim:output[fea].extend([padding]*(dense_fea_dim-len(output[fea])))
    data=[]
    for fea in slots_fea:data.extend(output[fea])
    return data
```



```python
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
```



# 四、DeepFMLayer模型
完成了数据集的处理后，下一步将进行模型的搭建。根据之前的DeepFM模型理论分析，DeepFMLayer由FM和DNN两层组成，下面将分别完成两部分后，进行组合，并调用训练集进行训练。

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class DeepFMLayer(nn.Layer):
    #DeepFMLayer由FM和DNN两层组成
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes):
        super(DeepFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes
        #加载FM和DNN两部分
        self.fm = FM(sparse_feature_number, sparse_feature_dim,
                     dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                       dense_feature_dim, dense_feature_dim + sparse_num_field,
                       layer_sizes)
        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    #前向传播预测
    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order, feat_embeddings = self.fm(sparse_inputs,
                                                                 dense_inputs)
        y_dnn = self.dnn(feat_embeddings)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FM(nn.Layer):
    #FM层，负责抽取low-order特征
    def __init__(self,
            sparse_feature_number = 1000001, 
            sparse_feature_dim = 9,
            dense_feature_dim = 13,
            sparse_num_field = 26):
        super(FM, self).__init__()
        self.sparse_feature_number = sparse_feature_number # 1000001 
        self.sparse_feature_dim = sparse_feature_dim# 9
        self.dense_feature_dim = dense_feature_dim#13
        self.sparse_num_field = sparse_num_field# sparse_inputs_slots-1==>26
        self.layer_sizes = layer_sizes#  fc_sizes: [512, 256, 128, 32]
        
        # 一阶稀疏特征
        self.sparse_feature_oneOrderWeight=paddle.nn.Embedding(
            sparse_feature_number,
            1,
            padding_idx=0,
            sparse=True
        )
        ## 一阶连续特征
        self.dense_feature_oneOrderWeight=paddle.create_parameter(
            [dense_feature_dim],
            "float32"
        )
        # 二阶特征
        self.sparse_latent_vecs=paddle.nn.Embedding(
            sparse_feature_number,
            embedding_dim,
            padding_idx=0,
            sparse=True
        )
        self.dense_latent_vecs=paddle.create_parameter(
            [1,dense_feature_dim,embedding_dim],
            "float32"
        )

    def forward(self,sparse_feature,dense_feature):
        # 一阶特征   

        '''
        计算一阶特征: y_1order = 0 + w*x
        input [batchsize,field_num]
        embed [batchsize,field_num,embedDim]
        sum out axis=1:[batchsize,embedDim]
        '''
        # 稀疏特征查表获得w*x  <- w*1 <- w <- lookup Embedding Table
        sparse_wx=self.sparse_feature_oneOrderWeight(sparse_feature)# [batchsize,sparse_field_num,1]
        # 连续特征向量内积w*x
        dense_wx=paddle.multiply(dense_feature,self.dense_feature_oneOrderWeight)  # [batchsize,dense_feature_dim]
        dense_wx=paddle.unsqueeze(dense_wx, axis=2)# [batchsize,dense_feature_dim,1]

        y_pred_first_order=paddle.sum(sparse_wx,axis=1)+paddle.sum(dense_wx,axis=1)# [batchsize,dense_feature_dim,1]---> [batchsize,1]
        
        # 二阶特征交叉
        '''
        y_2order=\sum{<Vi,Vj>xi xj}
        优化后计算公式为：
        vi,j * xi的平方和 减去 vi,j * vi 的和的平方，再取1/2   
        '''
        #稀疏特征查表: vij*xi<-vij *1
        sparse_vx= self.sparse_latent_vecs(sparse_feature) # [batchsize,sparse_field_num,embed_dim]
        '''
        连续特征矩阵乘法：
        
        dense_fea: [batchsize,dense_fea_dim,1]
        dense_latent_vecs:[1,dense_fea_dim,embed_dim]
        vij*xi <-  广播逐元素乘法（dense_fea，dense_latent_vecs）  #[batchsize,dense_fea_dim,embed_dim]
        '''
        dense_x=paddle.unsqueeze(dense_feature,axis=2) # [batchsize,dense_fea_dim]->[batchsize,dense_fea_dim,1]
        dense_vx=paddle.multiply(dense_x,self.dense_latent_vecs)#[batchsize,dense_fea_dim,embed_dim]
        
        concat_vx=paddle.concat([sparse_vx,dense_vx],axis=1)#[batchsize,sparse_field_num+dense_fea_dim,embed_dim]
        embedding=concat_vx
        #平方的和
        concat_vx_square=paddle.square(concat_vx)#[batchsize,sparse_field_num+dense_fea_dim,embed_dim]
        square_sum=paddle.sum(concat_vx_square,axis=1)#[batchsize,embed_dim]
        #和的平方
        concat_vx_sum=paddle.sum(concat_vx,axis=1)#[batchsize,embed_dim]
        sum_square=paddle.square(concat_vx_sum)#[batchsize,embed_dim]

        y_pred_second_order=0.5*(paddle.sum(sum_square-square_sum,axis=1))#[batchsize,1]
        y_pred_second_order=paddle.unsqueeze(y_pred_second_order,axis=1)
        return y_pred_first_order,y_pred_second_order,embedding


class DNN(paddle.nn.Layer):
    #DNN层，负责抽取high-order特征
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
    #利用FM模型的隐特征向量作为网络权重初始化来获得子网络输出向量
        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
    #得到输入层到embedding层该神经元相连的五条线的权重
    #前向传播反馈
    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn
```



# 五、模型训练
现在已经拥有了处理好的训练集和设计好的模型，下一步将利用数据集对模型进行训练。

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np

#模型参数
sparse_feature_number = 1000001 # 1000001 离散特征数
embedding_dim = 9# 9 嵌入层维度
dense_feature_dim = 13#13 稠密特征维度
sparse_num_field = 26# sparse_inputs_slots-1==>26 稀疏特征维度
layer_sizes = [512, 256, 128, 32]#  fc_sizes: [512, 256, 128, 32] 隐藏层数量

#训练参数
epochs = 2
batchsize=50
learning_rate=1e-3


def train(
    deepFM_model,
    deepFM_Dataset,
    batchnum,
    optimizer,
    sparse_feature_number = 1000001,
    embedding_dim = 9,
    dense_feature_dim = 13,
    sparse_num_field = 26,
    layer_sizes = [512, 256, 128, 32],

    epochs = 1,
    batchsize=500,
    learning_rate=1e-3):
    lossFunc=F.binary_cross_entropy
    for epoch in range(epochs):
        for batchidx in range(batchnum):
            #加载训练数据
            data=deepFM_Dataset.getNextBatchData()
            label_data = paddle.to_tensor(data[0],dtype='float32')#[batchsize,]
            label_data=paddle.unsqueeze(label_data,axis=1)#[batchsize,1]
            #得到稀疏/稠密特征
            sparse_feature = paddle.to_tensor(data[1], dtype='int64')
            dense_feature=paddle.to_tensor(data[2], dtype='float32')
            #得到预测值，为了得到每条样本分属于正负样本的概率，将预测结果和1-predict合并起来得到predicts，以便接下来计算auc
            predicts1 = deepFM_model(sparse_feature,dense_feature)#[batchsize,1]
            predicts0=1-predicts1#[batchsize,1]
            predicts=paddle.concat([predicts0,predicts1],axis=1)
            #计算auc指标
            auc = paddle.metric.Auc()
            auc.update(preds=predicts,labels=label_data)
            loss = lossFunc(predicts1, label_data)
            loss.backward()
            if batchidx % (batchnum//220) == 0:
                print("processing:{}%".format(100*batchidx/batchnum))
                print("label data 0-num: {0}  1-num:{1}".format( np.sum(data[0]<0.5),np.sum(data[0]>0.5) ) )
                print("epoch: {}, batch_id: {}, loss : {}, auc: {}".format(epoch, batchidx, loss.numpy(),auc.accumulate()))
                
            adam.step()
            adam.clear_grad()




epochs = 1
learning_rate=1e-3
trainFilePath='./work/slot_train_data_full'
trainFilesLineNum=40000000
#trainFilesLineNum=200000
batchsize=2000
trainBatchNum=trainFilesLineNum//batchsize

deepFM_TrainDataset=DeepFM_Dataset(batchsize,trainFilePath)
deepFM_model=DeepFMLayer(sparse_feature_number, embedding_dim, dense_feature_dim, sparse_num_field, layer_sizes)
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=deepFM_model.parameters())# Adam优化器

train(deepFM_model,deepFM_TrainDataset,epochs=epochs,batchsize=batchsize,batchnum=trainBatchNum,learning_rate=learning_rate,optimizer=adam)
```



# 六、训练过程可视化

```python
#此段代码不在AI Studio中运行，本地运行，读取上面的训练日志，画图可视化
import matplotlib.pyplot as plt
import xlrd
#plt的字体选择中文四黑
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 打开一个workbook
workbook = xlrd.open_workbook(r'1.xlsx')

# 抓取所有sheet页的名称
worksheets = workbook.sheet_names()
print('worksheets is %s' % worksheets)

# 定位到mySheet
mySheet = workbook.sheet_by_name(u'Sheet1')

# get datas
loss = mySheet.col_values(1)
print(loss)
time = mySheet.col(0)
print('time1',time)
time = [x.value for x in time]
print('time2',time)


#去掉标题行
loss.pop(0)
time.pop(0)

# declare a figure object to plot
fig = plt.figure(1)

# plot loss
plt.plot(time,loss)

plt.title('损失度loss随训练完成度变化曲线')
plt.ylabel('loss')
plt.xticks(range(0,1))
plt.show()
```

训练时，损失度loss随完成率曲线如下图：

![](https://ai-studio-static-online.cdn.bcebos.com/2d11650acc594ef38779153f9fc43842d901268b7cff472681509ddb3785fda0)



# 七、模型保存
将训练完成的模型保存下来。

```python
# save
#paddle.save(deepFM_model.state_dict(), "./model/deepFM_model.pdparams")
#paddle.save(adam.state_dict(), "./model/adam.pdopt")

layer_state_dict = paddle.load("./model/deepFM_model.pdparams")
opt_state_dict = paddle.load("./model/adam.pdopt")

testDeepFM_model=DeepFMLayer(sparse_feature_number = 1000001, sparse_feature_dim = 9,
                 dense_feature_dim = 13, sparse_num_field = 26, layer_sizes = [512, 256, 128, 32])
testAdam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=testDeepFM_model.parameters())# Adam优化器

testDeepFM_model.set_state_dict(layer_state_dict)
testAdam.set_state_dict(opt_state_dict)
```



# 八、模型预测
调用刚刚训练好的模型，对预测集进行预测。
输出预测为正/负样本的预测值以及真实值，最终给出这一批预测的loss和auc

```python
testFileLineNum=1788219
testFilePath='./work/slot_test_data_full'
batchsize=400
testBatchNum=testFileLineNum//batchsize
deepFM_TestDataset=DeepFM_Dataset(batchsize,testFilePath)

def predict(deepFM_model,deepFM_Dataset,batchnum):
    for batchidx in range(batchnum):
        #加载数据
        data=deepFM_Dataset.getNextBatchData()
        label_data = paddle.to_tensor(data[0],dtype='float32')#[batchsize,]
        label_data=paddle.unsqueeze(label_data,axis=1)#[batchsize,1]
        #得到特征
        sparse_feature = paddle.to_tensor(data[1], dtype='int64')
        dense_feature=paddle.to_tensor(data[2], dtype='float32')
        #得到预测值，为了得到每条样本分属于正负样本的概率，将预测结果和1-predict合并起来得到predicts，以便接下来计算auc
        predicts1 = deepFM_model(sparse_feature,dense_feature)#[batchsize,1]
        predicts0=1-predicts1#[batchsize,1]
        predicts=paddle.concat([predicts0,predicts1],axis=1)
        #计算auc
        auc = paddle.metric.Auc()
        auc.update(preds=predicts,labels=label_data)
        loss = F.binary_cross_entropy(predicts1, label_data)
        
        if batchidx % (batchnum//20)==0:
            print(paddle.concat([predicts[:4,],label_data[:4,]],axis=1).numpy())
            print("batchidx:{} loss:{} auc:{}".format(batchidx,loss.numpy(),auc.accumulate()))

predict(testDeepFM_model,deepFM_TestDataset,testBatchNum)
```



# 九、结果分析



| 模型   | auc  | loss | batch_size | epoch_num | 训练时间 |
| ------ | ---- | ---- | ---------- | --------- | -------- |
| DeepFM | 0.74 | 0.47 | 2000       | 1         | 1.5小时  |