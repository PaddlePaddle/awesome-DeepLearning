#coding:utf-8
import os

import paddle
import paddlehub as hub


class DemoDataset(paddle.io.Dataset):
    def __init__(self, transforms, num_classes=4, mode='train'):	
        # 数据集存放位置
        self.dataset_dir = "./dataset/peach-classification"  #dataset_dir为数据集实际路径，需要填写全路径
        self.transforms = transforms
        self.num_classes = num_classes
        self.mode = mode

        if self.mode == 'train':
            self.file = 'train_list.txt'
        elif self.mode == 'test':
            self.file = 'test_list.txt'
        else:
            self.file = 'validate_list.txt'
        
        self.file = os.path.join(self.dataset_dir , self.file)
        self.data = []
        
        with open(self.file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    self.data.append(line)
            
    def __getitem__(self, idx):
        img_path, grt = self.data[idx].split(' ')
        img_path = os.path.join(self.dataset_dir, img_path)
        im = self.transforms(img_path)
        return im, int(grt)


    def __len__(self):
        return len(self.data)
