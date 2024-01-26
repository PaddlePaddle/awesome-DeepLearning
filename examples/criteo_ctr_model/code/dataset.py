import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class CriteoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, min_count = 8):
        index_data, value_data, labels, feature_size, field_size = self.process_data(csv_path, min_count)        
        self.index_data  = index_data
        self.value_data = value_data
        self.labels = labels
        self.feature_size = feature_size
        self.field_size = field_size

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        return self.index_data[index], self.value_data[index], self.labels[index]
    
    def process_data(self, csv_path, min_count):
        #读取数据
        df = pd.read_csv(csv_path, header=None, sep='\t')
        num_cols = [f'I{i+1}' for i in range(13)]
        cate_cols = [f'C{i+1}' for i in range(26)]
        columns = ['label'] + num_cols + cate_cols
        df.columns = columns
    
        #数值特征 min-max归一化
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
        #类别特征 最少出现的次数
        le = LabelEncoder()
        for col in num_cols:
            df[col].fillna(0, inplace=True)

        for cate_col in cate_cols:
            df[cate_col].fillna("", inplace=True)
            df[cate_col] = le.fit_transform(df[cate_col])
            #统计field下 特征出现的频次
            val_count_dict = df[cate_col].value_counts()
            #频次 小于等于 最少出现频次的 置为特定的特征值
            df[cate_col] = df[cate_col].apply(lambda x: x if val_count_dict[x] >= min_count else -1)
        
        feat_index_dict = {}
        total_index = 0
        for col in df:
            if col in num_cols:
                feat_index_dict[col] = total_index
                total_index += 1
            elif col in cate_cols:
                vals = df[col].unique()
                feat_index_dict[col] = dict(zip(vals, range(total_index, total_index + len(vals))))
                total_index += len(vals)
            else:
                pass
        
        #数值型特征不变，类别型特征的value都置为1(因为one-hot的关系)
        value_df = df[num_cols + cate_cols]
        value_df[cate_cols] = 1
        index_df = df[num_cols + cate_cols]
        for col in num_cols:
            index_df[col] = feat_index_dict[col]
        for col in cate_cols:
            index_df[col] = df[col].map(feat_index_dict[col])
        labels = df['label'].values
        return index_df.to_numpy(), value_df.to_numpy().astype(np.float32), labels, total_index, len(num_cols + cate_cols)