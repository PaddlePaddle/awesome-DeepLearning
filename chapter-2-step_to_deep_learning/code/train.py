# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

from datasets.generate import MnistDataset
from nets import fnn, logistic
from visualdl import LogWriter
import paddle
import paddle.nn.functional as F
import numpy as np


class Trainer(object):
    def __init__(self, model_path, model, optimizer, summary_dir=None):
        self.model_path = model_path
        self.summary_dir = summary_dir
        self.model = model
        self.optimizer = optimizer
        self.summary_writer = self.get_summary_writer()
        self.global_step = 0

    def get_summary_writer(self):
        if self.summary_dir is None:
            return None
        else:
            return LogWriter(self.summary_dir)

    def update_summary(self, **kwargs):
        if self.summary_writer is None:
            pass
        else:
            for name in kwargs:
                self.summary_writer.add_scalar(tag=name, step=self.global_step, value=kwargs[name])

    def save(self):
        paddle.save(self.model.state_dict(), self.model_path)

    def val_epoch(self, datasets):
        self.model.eval()
        acc = list()
        for batch_id, data in enumerate(datasets()):
            images, labels = data
            pred = self.model(images)
            pred = paddle.argmax(pred, axis=-1)  # 取 pred 中得分最高的索引作为分类结果
            res = paddle.equal(pred, labels)
            res = paddle.cast(res, dtype='float32')
            acc.extend(res.numpy())  # 追加
        acc = np.array(acc).mean()
        return acc

    def train_step(self, data):
        images, labels = data

        # 前向计算的过程
        predicts = self.model(images)

        # 计算损失
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)

        # 后向传播，更新参数的过程
        avg_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return avg_loss

    def train_epoch(self, datasets, epoch):
        self.model.train()
        for batch_id, data in enumerate(datasets()):
            loss = self.train_step(data)
            self.update_summary(train_loss=loss.numpy())
            self.global_step += 1
            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

    def train(self, train_datasets, val_datasets, epochs):
        for i in range(epochs):
            self.train_epoch(train_datasets, i)
            train_acc = self.val_epoch(train_datasets)
            val_acc = self.val_epoch(val_datasets)
            self.update_summary(train_acc=train_acc, val_acc=val_acc)
            print("epoch_id: {}, train acc is: {}, val acc is {}".format(i, train_acc, val_acc))
        self.save()


def main():
    epochs = 10
    lr = 0.1
    model_path = './mnist.pdparams'

    train_dataset = MnistDataset(mode='train')
    train_loader = paddle.io.DataLoader(train_dataset,
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=4)

    val_dataset = MnistDataset(mode='val')
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=128)

    # model = fnn.MNIST()
    model = logistic.MNIST()
    # opt = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())
    opt = paddle.optimizer.SGD(learning_rate=lr,
                               weight_decay=paddle.regularizer.L2Decay(coeff=5e-4),
                               parameters=model.parameters())

    trainer = Trainer(
        model_path=model_path,
        model=model,
        optimizer=opt
    )

    trainer.train(train_datasets=train_loader, val_datasets=val_loader, epochs=epochs)


if __name__ == '__main__':
    main()
