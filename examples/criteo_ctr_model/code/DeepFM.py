import time
import gc
import joblib
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

#local code
import util
from dataset import CriteoDataset


class DeepFM(torch.nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, dnn_hidden = [128, 64]):
        super(DeepFM, self).__init__()
        self.shared_emb = torch.nn.Embedding(feature_size, embedding_size)
        self.fm_first_order_weight_emb = torch.nn.Embedding(feature_size, 1)
        self.fm_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.dnn_layer_1 = torch.nn.Linear(field_size*embedding_size, dnn_hidden[0])
        self.dnn_layer_2 = torch.nn.Linear(dnn_hidden[0], dnn_hidden[1])
        self.dnn_layer_3 = torch.nn.Linear(dnn_hidden[1], 1)
        
    #idx和vals的shape = (batch_size, field_size)
    def forward(self, idxs, vals):
        #dnn部分
        shared_emb = self.shared_emb(idxs) * vals.unsqueeze(-1)
        x = torch.flatten(shared_emb, start_dim=-2)
        x = self.dnn_layer_1(x)
        x = F.relu(x)
        x = self.dnn_layer_2(x)
        x = F.relu(x)
        dnn_out = self.dnn_layer_3(x)

        #fm 二次型
        tmp = torch.sum(shared_emb, dim=1, keepdim=True)
        square_of_sum = tmp * tmp
        sum_of_square = torch.sum(shared_emb * shared_emb, dim=1, keepdim=True)
        fm_second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=-1, keepdim=False)

        #fm 一次项
        fm_first_order_weight = self.fm_first_order_weight_emb(idxs)
        fm_first_order = torch.sum(fm_first_order_weight.squeeze(-1) * vals, dim=-1, keepdim=True)

        fm_out = fm_second_order + fm_first_order + self.fm_bias
        return torch.sigmoid(dnn_out + fm_out)



batch_size = 512
embedding_size = 32
dnn_hidden = [300, 300]
lr = 0.001
weight_l2 = 1e-4
epoch = 10
# device = torch.device('cuda:0')
device = torch.device('cpu')

dataset = CriteoDataset('../data/train_100w.txt')
# with open('../store/criteo_dataset_100w.jb', 'wb') as file:
#     joblib.dump(dataset, file)

# dataset = joblib.load('../store/criteo_dataset_100w.jb')

dataset_num = len(dataset)
train_num = int(dataset_num * 0.8)
valid_num = int(dataset_num * 0.1)
test_num = dataset_num - train_num - valid_num
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_num, valid_num, test_num))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=32)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32)

model = DeepFM(dataset.feature_size, dataset.field_size, embedding_size, dnn_hidden).to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_l2)
for epoch_i in range(epoch):
    util.train(model, optimizer, train_data_loader, criterion, device, verbose=200)
    auc = util.test(model, valid_data_loader, device)
    print('epoch:', epoch_i, 'validation: auc:', auc)
auc = util.test(model, test_data_loader, device)
print('test auc:', auc)
