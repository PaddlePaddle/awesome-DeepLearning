#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[5]:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class DNNLayer(nn.Layer):
    def __init__(self):
        super(DNNLayer, self).__init__()
        self.layer1 = 39
        self.layer2 = 64
        self.layer3 = 90
        self.layer4 = 32
        self.fclayer1 = nn.Linear(in_features=self.layer1,out_features=self.layer2)
        self.fclayer2 = nn.Linear(in_features=self.layer2,out_features=self.layer3) 
        self.fclayer3 = nn.Linear(in_features=self.layer3,out_features=self.layer4)
        self.fclayer4 = nn.Linear(in_features=self.layer4,out_features=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fclayer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fclayer2(x)
        x = F.relu(x)
        x = self.fclayer3(x)
        x = F.relu(x)
        output = self.fclayer4(x)
        return output


# In[6]:


from __future__ import print_function
import numpy as np

from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.init()

    def init(self):
        from operator import mul
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        full_lines = []
        self.data = []
        output_list = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    line = l.strip().split(" ")
                    output = [(i, []) for i in self.slots]
                    for i in line:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            feasign = int(slot_feasign[1])
                        else:
                            feasign = float(slot_feasign[1])
                        output[self.slot2index[slot]][1].append(feasign)
                        self.visit[slot] = True
                    for i in self.visit:
                        slot = i
                        if not self.visit[slot]:
                            if i in self.dense_slots:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding] *
                                    self.dense_slots_shape[self.slot2index[i]])
                            else:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding])
                        else:
                            self.visit[slot] = False
                    # sparse
                    
                    for key, value in output[:-1]:
                        # output_list.append(np.array(value).astype('int64'))
                        output_list.append((value[0]))
                    # dense
                    # output_list.append(
                        # np.array(output[-1][1]).astype("float32"))
                    for k in output[-1][1]:
                       output_list.append(k)
                    # list
                    # print(output)
        return output_list
rec = RecDataset(["data/train_data.txt"],1)
rec.init()
output_list = rec.__iter__()
# print(output_list)
output_set = np.array(output_list)
output_set = np.reshape(output_set,[80,40])
print(output_set.shape)


# In[94]:


#训练函数
from paddle.metric import Accuracy
input_dim =39
layer1_output = 64
layer2_output = 80
layer3_output = 32
data_num=80
# model =DNNLayer(input_dim,layer1_output,layer2_output,layer3_output)
val_acc_history = []
val_loss_history = []
train_set = output_set[:,1:]
# train_set.dtype='float32'
print(train_set.shape)
label = np.reshape(output_set[:,0],[data_num,1])
label = np.float64(label)
print(label.shape)
# train_dataset = paddle.empty(shape=[data_num,2])

# train_dataset = np.hstack((train_set,label))
# train_dataset = paddle.to_tensor(train_set)
# for i in range(data_num):
#     train_dataset[i,0]=paddle.to_tensor(train_set[i,:],dtype='float32')
#     train_dataset[i,1]=paddle.to_tensor(label[i,0])


import numpy as np
from paddle.io import Dataset

# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples,train_set,label_set):
        self.num_samples = num_samples
        self.train_set = train_set
        self.label_set = label_set
    def __getitem__(self, idx):
        example = paddle.to_tensor(self.train_set[idx,:],dtype='float32')
        label = paddle.to_tensor(self.label_set[idx,0],dtype='int64')
        return example, label

    def __len__(self):
        return self.num_samples

dataset = RandomDataset(80,train_set,label)
for i in range(2):
    print(dataset[:][1])


# In[4]:


def train(epochs):
    print('start training ... ')

    model = paddle.Model(DNNLayer())   # 用Model封装模型
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
    # 配置模型
    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss(),
        Accuracy()
    )
    model.fit(dataset,
        # dataset[:][1],
        epochs=epochs,
        batch_size=8,
        verbose=1
        )
train(10)


# In[1]:


import sys
import paddle.fluid.incubate.data_generator as dg
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import Counter
import os


class CriteoDataset(dg.MultiSlotDataGenerator):
    def setup(self, feat_dict_name):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46,
            231, 4008, 7393
        ]
        self.cont_diff_ = [
            self.cont_max_[i] - self.cont_min_[i]
            for i in range(len(self.cont_min_))
        ]
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)
        self.feat_dict_ = pickle.load(open(feat_dict_name, 'rb'))

    def _process_line(self, line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []
        for idx in self.continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[idx])
                feat_value.append(
                    (float(features[idx]) - self.cont_min_[idx - 1]) /
                    self.cont_diff_[idx - 1])
        for idx in self.categorical_range_:
            if features[idx] == '' or features[idx] not in self.feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[features[idx]])
                feat_value.append(1.0)
        label = [int(features[0])]
        return feat_idx, feat_value, label

    def test(self, filelist):
        def local_iter():
            for fname in filelist:
                with open(fname.strip(), 'r') as fin:
                    for line in fin:
                        feat_idx, feat_value, label = self._process_line(line)
                        yield [feat_idx, feat_value, label]

        return local_iter

    def generate_sample(self, line):
        def data_iter():
            feat_idx, feat_value, label = self._process_line(line)
            yield [('feat_idx', feat_idx), ('feat_value', feat_value), ('label',
                                                                        label)]

        return data_iter


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    criteo_dataset = CriteoDataset()
    if len(sys.argv) <= 1:
        sys.stderr.write("feat_dict needed for criteo reader.")
        exit(1)
    criteo_dataset.setup(sys.argv[1])
    criteo_dataset.run_from_stdin()


# In[2]:


import logging
import numpy as np
import pickle

# disable gpu training for this example 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import paddle
import paddle.fluid as fluid

from args import parse_args
from criteo_reader import CriteoDataset
from network_conf import ctr_deepfm_model
import utils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)


def infer():
    args = parse_args()

    place = fluid.CPUPlace()
    inference_scope = fluid.Scope()

    test_files = [
        os.path.join(args.test_data_dir, x)
        for x in os.listdir(args.test_data_dir)
    ]
    criteo_dataset = CriteoDataset()
    criteo_dataset.setup(args.feat_dict)
    test_reader = fluid.io.batch(
        criteo_dataset.test(test_files), batch_size=args.batch_size)

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    cur_model_path = os.path.join(args.model_output_dir,
                                  'epoch_' + args.test_epoch)

    with fluid.scope_guard(inference_scope):
        with fluid.framework.program_guard(test_program, startup_program):
            loss, auc, data_list, auc_states = ctr_deepfm_model(
                args.embedding_size, args.num_field, args.num_feat,
                args.layer_sizes, args.act, args.reg)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=data_list, place=place)
            main_program = fluid.default_main_program()
            fluid.load(main_program, cur_model_path, exe)
            for var in auc_states:  # reset auc states
                set_zero(var.name, scope=inference_scope, place=place)

            loss_all = 0
            num_ins = 0
            for batch_id, data_test in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program,
                                            feed=feeder.feed(data_test),
                                            fetch_list=[loss.name, auc.name])
                num_ins += len(data_test)
                loss_all += loss_val
                logger.info('TEST --> batch: {} loss: {} auc_val: {}'.format(
                    batch_id, loss_all / num_ins, auc_val))

            print(
                'The last log info is the total Logloss and AUC for all test data. '
            )


def set_zero(var_name,
             scope=fluid.global_scope(),
             place=fluid.CPUPlace(),
             param_type="int64"):
    """
    Set tensor of a Variable to zero.
    Args:
        var_name(str): name of Variable
        scope(Scope): Scope object, default is fluid.global_scope()
        place(Place): Place object, default is fluid.CPUPlace()
        param_type(str): param data type, default is int64
    """
    param = scope.var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype(param_type)
    param.set(param_array, place)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    utils.check_version()
    infer()


# In[3]:


from args import parse_args
import os
import paddle.fluid as fluid
import sys
from network_conf import ctr_deepfm_model
import time
import numpy
import pickle
import utils


def train():
    args = parse_args()
    # add ce
    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    print('---------- Configuration Arguments ----------')
    for key, value in args.__dict__.items():
        print(key + ':' + str(value))

    if not os.path.isdir(args.model_output_dir):
        os.mkdir(args.model_output_dir)

    loss, auc, data_list, auc_states = ctr_deepfm_model(
        args.embedding_size, args.num_field, args.num_feat, args.layer_sizes,
        args.act, args.reg)
    optimizer = fluid.optimizer.SGD(
        learning_rate=args.lr,
        regularization=fluid.regularizer.L2DecayRegularizer(args.reg))
    optimizer.minimize(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(data_list)
    pipe_command = 'python criteo_reader.py {}'.format(args.feat_dict)
    dataset.set_pipe_command(pipe_command)
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(args.num_thread)
    train_filelist = [
        os.path.join(args.train_data_dir, x)
        for x in os.listdir(args.train_data_dir)
    ]

    print('---------------------------------------------')
    for epoch_id in range(args.num_epoch):
        start = time.time()
        dataset.set_filelist(train_filelist)
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[loss, auc],
            fetch_info=['epoch %d batch loss' % (epoch_id + 1), "auc"],
            print_period=1000,
            debug=False)
        model_dir = os.path.join(args.model_output_dir,
                                 'epoch_' + str(epoch_id + 1))
        sys.stderr.write('epoch%d is finished and takes %f s\n' % (
            (epoch_id + 1), time.time() - start))
        main_program = fluid.default_main_program()
        fluid.io.save(main_program, model_dir)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    utils.check_version()
    train()

