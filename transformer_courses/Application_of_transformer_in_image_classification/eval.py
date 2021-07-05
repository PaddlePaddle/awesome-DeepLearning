# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse
import numpy as np
import paddle
import paddle.nn.functional as F
from model import VisionTransformer, DistilledVisionTransformer
from dataset import ImageNetDataset


def eval(args):
    # 开启0号GPU
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('start evaluation .......')
    # 实例化模型
    if args.model == 'ViT':
        model = VisionTransformer(
                patch_size=16,
                class_dim=1000,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6)
        params_file_path = "model_file/ViT_base_patch16_384_pretrained.pdparams"
    else:
        model = DistilledVisionTransformer(
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                epsilon=1e-6)
        params_file_path="model_file/DeiT_base_distilled_patch16_384_pretrained.pdparams"

    # 加载模型参数
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)

    model.eval()

    VAL_FILE_LIST = os.path.join(args.data, 'val_list.txt')

    # 创建数据读取类
    val_dataset = ImageNetDataset(args.data, VAL_FILE_LIST)

    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=2, num_workers=1, drop_last=True)

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(val_loader()):
        x_data, y_data = data
        y_data = y_data.reshape([-1, 1])
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        logits = model(img)
        # 多分类，使用softmax计算预测概率
        pred = F.softmax(logits)
        # 计算交叉熵损失函数
        loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(logits, label)
        # 计算准确率
        acc = paddle.metric.accuracy(pred, label)

        acc_set.append(acc.numpy())
        avg_loss_set.append(loss.numpy())
    print("[validation] accuracy/loss: {}/{}".format(np.mean(acc_set), np.mean(avg_loss_set)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Transformer based on ImageNet')
    parser.add_argument('--model', type=str, default='ViT', help='Transformer model')
    parser.add_argument('--data', type=str, default='data/ILSVRC2012_val', help='Data dir')
    args = parser.parse_args()
    eval(args)
