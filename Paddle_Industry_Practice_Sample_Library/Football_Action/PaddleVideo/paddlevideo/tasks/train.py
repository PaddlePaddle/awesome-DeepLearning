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

import os.path as osp
import time

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddlevideo.utils import (add_profiler_step, build_record, get_logger,
                               load, log_batch, log_epoch, mkdir, save)

from ..loader.builder import build_dataloader, build_dataset
from ..metrics.ava_utils import collect_results_cpu
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN


def train_model(cfg,
                weights=None,
                parallel=True,
                validate=True,
                amp=False,
                max_iters=None,
                use_fleet=False,
                profiler_options=None):
    """Train model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path for finetuning.
        parallel (bool): Whether multi-cards training. Default: True.
        validate (bool): Whether to do evaluation. Default: False.
        amp (bool): Whether to use automatic mixed precision during training. Default: False.
        use_fleet (bool):
        profiler_options (str): Activate the profiler function Default: None.
    """
    if use_fleet:
        fleet.init(is_collective=True)

    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get('batch_size', 8)
    valid_batch_size = cfg.DATASET.get('valid_batch_size', batch_size)

    use_gradient_accumulation = cfg.get('GRADIENT_ACCUMULATION', None)
    if use_gradient_accumulation and dist.get_world_size() >= 1:
        global_batch_size = cfg.GRADIENT_ACCUMULATION.get(
            'global_batch_size', None)
        num_gpus = dist.get_world_size()

        assert isinstance(
            global_batch_size, int
        ), f"global_batch_size must be int, but got {type(global_batch_size)}"
        assert batch_size <= global_batch_size, f"global_batch_size must not be less than batch_size"

        cur_global_batch_size = batch_size * num_gpus  # The number of batches calculated by all GPUs at one time
        assert global_batch_size % cur_global_batch_size == 0, \
            f"The global batchsize must be divisible by cur_global_batch_size, but \
                {global_batch_size} % {cur_global_batch_size} != 0"

        cfg.GRADIENT_ACCUMULATION[
            "num_iters"] = global_batch_size // cur_global_batch_size
        # The number of iterations required to reach the global batchsize
        logger.info(
            f"Using gradient accumulation training strategy, "
            f"global_batch_size={global_batch_size}, "
            f"num_gpus={num_gpus}, "
            f"num_accumulative_iters={cfg.GRADIENT_ACCUMULATION.num_iters}")

    if cfg.get('use_npu'):
        places = paddle.set_device('npu')
    else:
        places = paddle.set_device('gpu')

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    valid_num_workers = cfg.DATASET.get('valid_num_workers', num_workers)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # 1. Construct model
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    if use_fleet:
        model = paddle.distributed_model(model)

    # 2. Construct dataset and dataloader
    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.train))
    train_dataloader_setting = dict(batch_size=batch_size,
                                    num_workers=num_workers,
                                    collate_fn_cfg=cfg.get('MIX', None),
                                    places=places)

    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)

    if validate:
        valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
        validate_dataloader_setting = dict(
            batch_size=valid_batch_size,
            num_workers=valid_num_workers,
            places=places,
            drop_last=False,
            shuffle=cfg.DATASET.get(
                'shuffle_valid',
                False)  #NOTE: attention lstm need shuffle valid data.
        )
        valid_loader = build_dataloader(valid_dataset,
                                        **validate_dataloader_setting)

    # 3. Construct solver.
    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(cfg.OPTIMIZER, lr, model=model)
    if use_fleet:
        optimizer = fleet.distributed_optimizer(optimizer)
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
    ###AMP###
    if amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=2.0**16,
                                       incr_every_n_steps=2000,
                                       decr_every_n_nan_or_inf=1)

    best = 0.0
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
            """Next two line of code only used in test_tipc,
            ignore it most of the time"""
            if max_iters is not None and i >= max_iters:
                break

            record_list['reader_time'].update(time.time() - tic)

            # Collect performance information when profiler_options is activate
            add_profiler_step(profiler_options)

            # 4.1 forward
            # AMP #
            if amp:
                with paddle.amp.auto_cast(custom_black_list={"reduce_mean"}):
                    outputs = model(data, mode='train')
                avg_loss = outputs['loss']
                if use_gradient_accumulation:
                    # clear grad at when epoch begins
                    if i == 0:
                        optimizer.clear_grad()
                    # Loss normalization
                    avg_loss /= cfg.GRADIENT_ACCUMULATION.num_iters
                    # Loss scaling
                    scaled = scaler.scale(avg_loss)
                    # 4.2 backward
                    scaled.backward()
                    # 4.3 minimize
                    if (i + 1) % cfg.GRADIENT_ACCUMULATION.num_iters == 0:
                        scaler.minimize(optimizer, scaled)
                        optimizer.clear_grad()
                else:  # general case
                    # 4.2 backward
                    scaled = scaler.scale(avg_loss)
                    scaled.backward()
                    # 4.3 minimize
                    scaler.minimize(optimizer, scaled)
                    optimizer.clear_grad()
            else:
                outputs = model(data, mode='train')
                avg_loss = outputs['loss']
                if use_gradient_accumulation:
                    # clear grad at when epoch begins
                    if i == 0:
                        optimizer.clear_grad()
                    # Loss normalization
                    avg_loss /= cfg.GRADIENT_ACCUMULATION.num_iters
                    # 4.2 backward
                    avg_loss.backward()
                    # 4.3 minimize
                    if (i + 1) % cfg.GRADIENT_ACCUMULATION.num_iters == 0:
                        optimizer.step()
                        optimizer.clear_grad()
                else:  # general case
                    # 4.2 backward
                    avg_loss.backward()
                    # 4.3 minimize
                    optimizer.step()
                    optimizer.clear_grad()

            # log record
            record_list['lr'].update(optimizer.get_lr(), batch_size)
            for name, value in outputs.items():
                if name in record_list:
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

        ips = "avg_ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

        def evaluate(best):
            model.eval()
            results = []
            record_list = build_record(cfg.MODEL)
            record_list.pop('lr')
            tic = time.time()
            if parallel:
                rank = dist.get_rank()
            #single_gpu_test and multi_gpu_test
            for i, data in enumerate(valid_loader):
                outputs = model(data, mode='valid')
                if cfg.MODEL.framework == "FastRCNN":
                    results.extend(outputs)

                #log_record
                if cfg.MODEL.framework != "FastRCNN":
                    for name, value in outputs.items():
                        if name in record_list:
                            record_list[name].update(value, batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        valid_batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, cfg.epochs, "val", ips)

            if cfg.MODEL.framework == "FastRCNN":
                if parallel:
                    results = collect_results_cpu(results, len(valid_dataset))
                if not parallel or (parallel and rank == 0):
                    eval_res = valid_dataset.evaluate(results)
                    for name, value in eval_res.items():
                        record_list[name].update(value, valid_batch_size)

            ips = "avg_ips: {:.5f} instance/sec.".format(
                valid_batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            if cfg.MODEL.framework == "FastRCNN" and (not parallel or
                                                      (parallel and rank == 0)):
                if record_list["mAP@0.5IOU"].val > best:
                    best = record_list["mAP@0.5IOU"].val
                    best_flag = True
                return best, best_flag

            # forbest2, cfg.MODEL.framework != "FastRCNN":
            for top_flag in ['hit_at_one', 'top1', 'rmse']:
                if record_list.get(top_flag):
                    if top_flag != 'rmse' and record_list[top_flag].avg > best:
                        best = record_list[top_flag].avg
                        best_flag = True
                    elif top_flag == 'rmse' and (
                            best == 0.0 or record_list[top_flag].avg < best):
                        best = record_list[top_flag].avg
                        best_flag = True

            return best, best_flag

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (epoch % cfg.PRECISEBN.preciseBN_interval
                                     == 0 or epoch == cfg.epochs - 1):
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))

        # 5. Validation
        if validate and (epoch % cfg.get("val_interval", 1) == 0
                         or epoch == cfg.epochs - 1):
            with paddle.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
                if model_name == "AttentionLstm":
                    logger.info(
                        f"Already save the best model (hit_at_one){best}")
                elif cfg.MODEL.framework == "FastRCNN":
                    logger.info(
                        f"Already save the best model (mAP@0.5IOU){int(best * 10000) / 10000}"
                    )
                elif cfg.MODEL.framework == "DepthEstimator":
                    logger.info(
                        f"Already save the best model (rmse){int(best * 10000) / 10000}"
                    )
                else:
                    logger.info(
                        f"Already save the best model (top1 acc){int(best * 10000) / 10000}"
                    )

        # 6. Save model and optimizer
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
