import re
import os
import time
import tarfile
import random
import argparse
import numpy as np
from functools import partial

import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.metric import Accuracy

from paddlenlp.datasets import MapDataset
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer
from paddlenlp.transformers.xlnet.modeling import XLNetPretrainedModel, XLNetForSequenceClassification

from utils import set_seed
from data_processor import IMDBDataset, convert_example


def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fct(logits, labels)
        losses.append(loss.detach().numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("eval loss: %f, acc: %s" % (np.average(losses), res))
    model.train()

def train(args):

    # 加载数据
    trainset=IMDBDataset(is_training=True)
    testset = IMDBDataset(is_training=False)

    # 封装成MapDataSet的形式
    train_ds = MapDataset(trainset, label_list=[0,1])
    test_ds = MapDataset(testset, label_list=[0,1])
    
    # 定义XLNet的Tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer = tokenizer,
        label_list = train_ds.label_list,
         max_seq_length= args.max_seq_length
    )

    # 构造train_data_loader 和 dev_data_loader
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size = args.batch_size, shuffle=True
    )

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, pad_right=False),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, pad_right=False),  # token_type
        Pad(axis=0, pad_val=0, pad_right=False),  # attention_mask
        Stack(dtype="int64" if train_ds.label_list else "float32"),  # label
    ): fn(samples)

    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_ds = MapDataset(testset)
    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)

    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    # 训练配置
    # 固定随机种子
    set_seed(args)

    # 设定运行环境
    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    num_classes = len(train_ds.label_list)
    model = XLNetForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=num_classes)

    #paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        model = paddle.DataParallel(model)

    # 设定lr_scheduler
    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = ceil(num_training_steps / len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # 制定优化器
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.max_grad_norm)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "layer_norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        grad_clip=clip,
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    # 模型训练
    metric = Accuracy()

    # 定义损失函数
    loss_fct = paddle.nn.loss.CrossEntropyLoss(
    ) if train_ds.label_list else paddle.nn.loss.MSELoss()

    global_step = 0
    tic_train = time.time()
    model.train()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, attention_mask, labels = batch
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                evaluate(model, loss_fct, metric, dev_data_loader)
                print("eval done total : %s s" % (time.time() - tic_eval))

                if (not paddle.distributed.get_world_size() > 1
                    ) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir, "%s_ft_model_%d"
                                              % (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                if global_step == num_training_steps:
                    exit(0)
                tic_train += time.time() - tic_eval


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Reading Comprehension based on ERNIE.")
    parser.add_argument("--model_name_or_path", type=str, default="xlnet-base-cased", help="the model you want to load.")
    parser.add_argument("--task_name", type=str, default="sst-2")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="the epochs of model training.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="the max_seq_length of input sequence.")
    parser.add_argument("--doc_stride", type=int, default=128, help="doc_stride when processing data.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size when model training.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon setting.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning_rate for model training.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm applying adjusting gradient.")
    parser.add_argument("--max_steps", type=int, default=-1, help="the max steps you want to train.")
    parser.add_argument("--logging_steps", type=int, default=100, help="how many steps to log info.")
    parser.add_argument("--save_steps", type=int, default=500, help="how many steps to save model.")
    parser.add_argument("--seed", type=int, default=43, help="random seed.")
    parser.add_argument("--device", type=str, default="gpu", help="cpu or gpu selection.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="the proportion of performing warmup in all training steps.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="the weight_decay of model parameters.")   
    parser.add_argument("--output_dir", type=str, default="./tmp", help="the path of saving model.") 

    args = parser.parse_args()

    train(args)
