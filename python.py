import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
 
data = pd.read_csv('train.csv')  # 查看数据
data.head()                      # 查看数据集形状
data.shape                       # 查看数据集数据类型
data.dtypes


datafile = './housing.data'
housing_data = np.fromfile(datafile, sep=' ')
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_num = len(feature_names)     # 从文件导入数据
housing_data = housing_data.reshape([housing_data.shape[0] // feature_num, feature_num])    # 将原始数据进行Reshape，变成[N, 14]这样的形状
features_max = housing_data.max(axis=0)
features_min = housing_data.min(axis=0)
features_avg = housing_data.sum(axis=0) / housing_data.shape[0]
BATCH_SIZE = 20
def feature_norm(input):
    f_size = input.shape
    output_features = np.zeros(f_size, np.float32)
    for batch_id in range(f_size[0]):
        for index in range(13):
            output_features[batch_id][index] = (input[batch_id][index] - features_avg[index]) / (features_max[index] - features_min[index])
    return output_features 
housing_features = feature_norm(housing_data[:, :13])      # 只对属性进行归一化
housing_data = np.c_[housing_features, housing_data[:, -1]].astype(np.float32)
features_np = np.array([x[:13] for x in housing_data],np.float32)
labels_np = np.array([x[-1] for x in housing_data],np.float32)
data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(data_np, columns=feature_names)
ratio = 0.8
offset = int(housing_data.shape[0] * ratio)
train_data = housing_data[:offset]
test_data = housing_data[offset:]# 将训练数据集和测试数据集按照8:2的比例分开
class Net(nn.Module):          #定义存储网路结构
    def __init__(self):
        super(Net,self).__init__()
        self.classify=nn.Sequential(      #nn 模块搭建网络
            nn.Linear(2,15),                    
            nn.ReLU(),
            nn.Linear(15,2),
            nn.Softmax(dim=1)
        )
    def forward(self,x):              #前向传播
        classification=self.classify(x)
        return classification


net=Net()            #训练网络
optimizer=torch.optim.SGD(net.parameters(),lr=0.03)     #优化器
loss_func=nn.CrossEntropyLoss()                #损失函数
for epoch in range(100):
    out=net(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
train_nums = []
train_costs = []

def draw_train_process(iters, train_costs):
    plt.title("training cost", fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.show()