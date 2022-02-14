# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import os.path as osp

import paddle
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import (AverageMeter, build_record, log_batch, log_epoch,
                               save, load, mkdir)
from paddlevideo.loader import TSN_Dali_loader, get_input_data
"""
We only supported DALI training for TSN model now.
"""


def train_dali(cfg, weights=None, parallel=True):
    """Train model entry

    Args:
    	cfg (dict): configuration.
        weights (str): weights path for finetuning.
    	parallel (bool): Whether multi-cards training. Default: True.

    """

    logger = get_logger("paddlevideo")
    batch_size = cfg.DALI_LOADER.get('batch_size', 8)
    places = paddle.set_device('gpu')
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # 1. Construct model
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dali dataloader
    train_loader = TSN_Dali_loader(cfg.DALI_LOADER).build_dali_reader()

    # 3. Construct solver.
    lr = build_lr(cfg.OPTIMIZER.learning_rate, None)
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}")
        resume_model_dict = load(filename + '.pdparams')
        resume_opt_dict = load(filename + '.pdopt')
        model.set_state_dict(resume_model_dict)
        optimizer.set_state_dict(resume_opt_dict)

    # Finetune:
    if weights:
        assert resume_epoch == 0, f"Conflict occurs when finetuning, please switch resume function off by setting resume_epoch to 0 or not indicating it."
        model_dict = load(weights)
        model.set_state_dict(model_dict)

    # 4. Train Model
    for epoch in range(0, cfg.epochs):
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue
        model.train()
        record_list = build_record(cfg.MODEL)
        tic = time.time()
        for i, data in enumerate(train_loader):
            data = get_input_data(data)
            record_list['reader_time'].update(time.time() - tic)
            # 4.1 forward
            outputs = model(data, mode='train')
            # 4.2 backward
            avg_loss = outputs['loss']
            avg_loss.backward()
            # 4.3 minimize
            optimizer.step()
            optimizer.clear_grad()

            # log record
            record_list['lr'].update(optimizer._global_learning_rate(),
                                     batch_size)
            for name, value in outputs.items():
                record_list[name].update(value, batch_size)

            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, cfg.epochs, "train", ips)

            # learning rate iter step
            if cfg.OPTIMIZER.learning_rate.get("iter_step"):
                lr.step()

        # learning rate epoch step
        if not cfg.OPTIMIZER.learning_rate.get("iter_step"):
            lr.step()

        ips = "ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (epoch % cfg.PRECISEBN.preciseBN_interval
                                     == 0 or epoch == cfg.epochs - 1):
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))

        # 5. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            save(
                optimizer.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1:05d}.pdopt"))
            save(
                model.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1:05d}.pdparams"))

    logger.info(f'training {model_name} finished')
