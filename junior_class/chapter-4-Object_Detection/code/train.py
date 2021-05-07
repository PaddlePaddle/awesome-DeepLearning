# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import paddle
import numpy as np
from dataset import TrainDataset
from net import YOLOv3

ANCHORS = [
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373,
    326
]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

IGNORE_THRESH = .7
NUM_CLASSES = 7


def get_lr(base_lr=0.0001, lr_decay=0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr)
    return learning_rate


if __name__ == '__main__':

    TRAINDIR = 'data/insects/train'
    TESTDIR = 'data/insects/test'
    VALIDDIR = 'data/insects/val'
    paddle.set_device("gpu:0")
    # 创建数据读取类
    train_dataset = TrainDataset(TRAINDIR, mode='train')
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        use_shared_memory=False)
    valid_loader = paddle.io.DataLoader(
        valid_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        use_shared_memory=False)
    model = YOLOv3(num_classes=NUM_CLASSES)  # 创建模型
    learning_rate = get_lr()
    opt = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=paddle.regularizer.L2Decay(0.0005),
        parameters=model.parameters())  # 创建优化器
    # opt = paddle.optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())

    MAX_EPOCH = 200
    for epoch in range(MAX_EPOCH):
        for i, data in enumerate(train_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)  # 前向传播，输出[P0, P1, P2]
            loss = model.get_loss(
                outputs,
                gt_boxes,
                gt_labels,
                gtscore=gt_scores,
                anchors=ANCHORS,
                anchor_masks=ANCHOR_MASKS,
                ignore_thresh=IGNORE_THRESH,
                use_label_smooth=False)  # 计算损失函数

            loss.backward()  # 反向传播计算梯度
            opt.step()  # 更新参数
            opt.clear_grad()
            if i % 10 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(time.time()))
                print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(
                    timestring, epoch, i, loss.numpy()))

        # save params of model
        if (epoch % 5 == 0) or (epoch == MAX_EPOCH - 1):
            paddle.save(model.state_dict(), 'yolo_epoch{}'.format(epoch))

        # 每个epoch结束之后在验证集上进行测试
        model.eval()
        for i, data in enumerate(valid_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)
            loss = model.get_loss(
                outputs,
                gt_boxes,
                gt_labels,
                gtscore=gt_scores,
                anchors=ANCHORS,
                anchor_masks=ANCHOR_MASKS,
                ignore_thresh=IGNORE_THRESH,
                use_label_smooth=False)
            if i % 1 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(time.time()))
                print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(
                    timestring, epoch, i, loss.numpy()))
        model.train()
