### DeepDM实现广告点击预测（criteo数据集）
## 一、 DeepFM算法的提出

由于DeepFM算法有效的结合了因子分解机与神经网络在特征学习中的优点：同时提取到低阶组合特征与高阶组合特征，所以越来越被广泛使用。

在DeepFM中，FM算法负责对一阶特征以及由一阶特征两两组合而成的二阶特征进行特征的提取；DNN算法负责对由输入的一阶特征进行全连接等操作形成的高阶特征进行特征的提取。

具有以下特点：

    1、结合了广度和深度模型的优点，联合训练FM模型和DNN模型，同时学习低阶特征组合和高阶特征组合。

    2、端到端模型，无需特征工程。

    3、DeepFM 共享相同的输入和 embedding vector，训练更高效。

    4、评估模型时，用到了一个新的指标“Gini Normalization”

## 二、DeepFM算法结构图

![](https://ai-studio-static-online.cdn.bcebos.com/dadd7b2b7aaf47dfa689fde0d4c5d81834249b7a838b4a389b7b7451b0908aa7)

其中，DeepFM的输入可由连续型变量和类别型变量共同组成，且类别型变量需要进行One-Hot编码。而正由于One-Hot编码，导致了输入特征变得高维且稀疏。

应对的措施是：针对高维稀疏的输入特征，采用Word2Vec的词嵌入（WordEmbedding）思想，把高维稀疏的向量映射到相对低维且向量元素都不为零的空间向量中。

实际上，这个过程就是FM算法中交叉项计算的过程，具体可参考我的另一篇文章：FM算法解析及Python实现 中5.4小节的内容。

由上面网络结构图可以看到，DeepFM 包括 FM和 DNN两部分，所以模型最终的输出也由这两部分组成：

下面，把结构图进行拆分。首先是FM部分的结构：

![](https://ai-studio-static-online.cdn.bcebos.com/8da4a46f26ea4a0da0737f3181bbc34c021c6e613a1a48fc98424c39699996af)

然后是DNN部分的结构：

![](https://ai-studio-static-online.cdn.bcebos.com/f9f15a083d1e46cea50fba175c3f5931724bcf7d2ddc46c585ac69fbe2cf4ec2)

这里DNN的作用是构造高维特征，且有一个特点：DNN的输入也是embedding vector。所谓的权值共享指的就是这里。

### 数据集
criteo是非常经典的点击率预估比赛）。训练集4千万行，特征连续型的有13个，类别型的26个，没有提供特征名称，样本按时间排序。测试集6百万行


```python
import paddle
from paddle import nn


class DeepFM(nn.Layer):
    def __init__(self, cate_fea_uniques,
                 num_fea_size=0,
                 emb_size=8,
                 hidden_dims=[256, 128],
                 num_classes=1,
                 dropout=[0.2, 0.2]):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size:
        :param hidden_dims:
        :param num_classes:
        :param dropout:
        '''
        super(DeepFM, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size

        # DeepFM
        # dense特征一阶表示
        if self.num_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.num_fea_size, 1)

        # sparse特征一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_uniques
        ])

        # sparse特征二阶表示
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques
        ])

        # DNN部分
        self.dense_linear = nn.Linear(self.num_fea_size, self.cate_fea_size * emb_size)  # # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()

        self.all_dims = [self.cate_fea_size * emb_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))

        self.dnn_linear = nn.Linear(hidden_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        """FM部分"""
        # 一阶  包含sparse_feature和dense_feature的一阶
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]  # sparse特征嵌入成一维度
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # torch.Size([2, 26])
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs, 1] 将sparse_feature通过全连接并相加整成一维度

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)   # 将dense_feature压到一维度
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]

        # 二阶
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # batch_size, sparse_feature_nums, emb_size
        # print(fm_2nd_concat_1d.size())   # torch.Size([2, 26, 8])

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # batch_size, emb_size
        square_sum_embed = sum_embed * sum_embed   # batch_size, emb_size

        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]

        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5   # batch_size, embed_size

        # 再求和
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # batch_size, 1

        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)   # [bs, n * emb_size]

        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))  # batch_size, sparse_feature_num * emb_size
            dnn_out = dnn_out + dense_out   # batch_size, sparse_feature_num * emb_size

        # 从sparse_feature_num * emb_size 维度 转为 sparse_feature_num * emb_size 再转为 256
        # print(self.all_dims)   # [208, 256, 128]
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        dnn_out = self.dnn_linear(dnn_out)   # batch_size, 1
        out = fm_1st_part + fm_2nd_part + dnn_out   # [bs, 1]
        out = self.sigmoid(out)
        return out
```


```python
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./train.txt", type=str)
    parser.add_argument('--Epochs', type=int, default=10)

    parser.add_argument('--train_batch_size', type=int, default=2, help="train batch size")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="eval batch size")

    parser.add_argument('--learning_rate', type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay")
    parser.add_argument('--n_gpu', type=int, default=0, help="n gpu")
    args = parser.parse_args(args=[])
    return args
```


```python
import os
import paddle
import time
from paddle import nn
import numpy as np
import pandas as pd
from paddle import optimizer
from paddle import to_tensor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from paddle.io import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

args
def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(valid_loader)):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            logits = model(cat_fea, num_fea)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        return cur_auc


def train_model(model):
    # 指定多gpu运行
    if torch.cuda.is_available():
        model.cuda()

    if torch.cuda.device_count() > 1:
        args.n_gpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)

    loss_fct = nn.BCELoss()
    optimizer = optimizer.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(args.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            pred = model(cat_fea, num_fea)
            # print(pred.size())  # torch.Size([2, 1])
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model(model)
        if cur_auc > best_auc:
            best_auc = cur_auc
            os.makedirs('./save_model', exist_ok=True)
            torch.save(model.state_dict(), './save_model/deepfm.bin')


if __name__ == '__main__':
    args = set_args()
    data = pd.read_csv(args.data_path)
    # print(data.shape)   # (600000, 40)

    # 数据预处理
    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    # 处理缺失值
    data[sparse_features] = data[sparse_features].fillna('-10086',)
    data[dense_features] = data[dense_features].fillna(0,)
    target = ['label']   

    # 将类别数据转为数字
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 将连续值归一化
    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)
    # print(data.shape)
    # print(data.head())

    train, valid = train_test_split(data, test_size=0.1, random_state=42)
    # print(train.shape)   # (540000, 40)
    # print(valid.shape)   # (60000, 40)
    train_dataset = TensorDataset(to_tensor(train[sparse_features].values),
                                  to_tensor(train[dense_features].values),
                                  to_tensor(train['label'].values))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)

    valid_dataset = TensorDataset(to_tensor(valid[sparse_features].values),
                                  to_tensor(valid[dense_features].values),
                                  to_tensor(valid['label'].values))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.eval_batch_size, shuffle=False)

    cat_fea_unique = [data[f].nunique() for f in sparse_features]

    model = DeepFM(cat_fea_unique, num_fea_size=len(dense_features))

    train_model(model)
```
