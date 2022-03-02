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
import paddle.distributed as dist

from ..loader.builder import build_dataloader, build_dataset
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import (AverageMeter, build_record, log_batch, log_epoch,
                               save, load, mkdir)
from paddlevideo.utils.multigrid import MultigridSchedule, aggregate_sub_bn_stats, subn_load, subn_save, is_eval_epoch


def construct_loader(cfg, places, validate, precise_bn, num_iters_precise_bn,
                     world_size):
    batch_size = cfg.DATASET.get('batch_size', 2)
    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.train))
    precise_bn_dataloader_setting = dict(
        batch_size=batch_size,
        num_workers=cfg.DATASET.get('num_workers', 0),
        places=places,
    )
    if precise_bn:
        cfg.DATASET.train.num_samples_precise_bn = num_iters_precise_bn * batch_size * world_size
        precise_bn_dataset = build_dataset(
            (cfg.DATASET.train, cfg.PIPELINE.train))
        precise_bn_loader = build_dataloader(precise_bn_dataset,
                                             **precise_bn_dataloader_setting)
        cfg.DATASET.train.num_samples_precise_bn = None
    else:
        precise_bn_loader = None

    if cfg.MULTIGRID.SHORT_CYCLE:
        # get batch size list in short cycle schedule
        bs_factor = [
            int(
                round((float(
                    cfg.PIPELINE.train.transform[1]['MultiCrop']['target_size'])
                       / (s * cfg.MULTIGRID.default_crop_size))**2))
            for s in cfg.MULTIGRID.short_cycle_factors
        ]
        batch_sizes = [
            batch_size * bs_factor[0],
            batch_size * bs_factor[1],
            batch_size,
        ]
        train_dataloader_setting = dict(
            batch_size=batch_sizes,
            multigrid=True,
            num_workers=cfg.DATASET.get('num_workers', 0),
            places=places,
        )
    else:
        train_dataloader_setting = precise_bn_dataloader_setting

    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)
    if validate:
        valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
        validate_dataloader_setting = dict(batch_size=batch_size,
                                           num_workers=cfg.DATASET.get(
                                               'num_workers', 0),
                                           places=places,
                                           drop_last=False,
                                           shuffle=False)
        valid_loader = build_dataloader(valid_dataset,
                                        **validate_dataloader_setting)
    else:
        valid_loader = None

    return train_loader, valid_loader, precise_bn_loader


def build_trainer(cfg, places, parallel, validate, precise_bn,
                  num_iters_precise_bn, world_size):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs.
    Returns:
        model: training model.
        optimizer: optimizer.
        train_loader: training data loader.
        val_loader: validatoin data loader.
        precise_bn_loader: training data loader for computing
            precise BN.
    """
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    train_loader, valid_loader, precise_bn_loader = \
        construct_loader(cfg,
                         places,
                         validate,
                         precise_bn,
                         num_iters_precise_bn,
                         world_size,
                         )

    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())

    return (
        model,
        lr,
        optimizer,
        train_loader,
        valid_loader,
        precise_bn_loader,
    )


def train_model_multigrid(cfg, world_size=1, validate=True):
    """Train model entry

    Args:
    	cfg (dict): configuration.
    	parallel (bool): Whether multi-card training. Default: True
        validate (bool): Whether to do evaluation. Default: False.

    """
    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    multi_save_epoch = [i[-1] - 1 for i in multigrid.schedule]

    parallel = world_size != 1
    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get('batch_size', 2)

    if cfg.get('use_npu'):
        places = paddle.set_device('npu')
    else:
        places = paddle.set_device('gpu')
    
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)
    local_rank = dist.ParallelEnv().local_rank
    precise_bn = cfg.get("PRECISEBN")
    num_iters_precise_bn = cfg.PRECISEBN.num_iters_preciseBN

    # 1. Construct model
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dataloader
    train_loader, valid_loader, precise_bn_loader = \
        construct_loader(cfg,
                         places,
                         validate,
                         precise_bn,
                         num_iters_precise_bn,
                         world_size,
                         )

    # 3. Construct optimizer
    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(
            output_dir,
            model_name + str(local_rank) + '_' + f"{resume_epoch:05d}")
        subn_load(model, filename, optimizer)

    # 4. Train Model
    best = 0.
    total_epochs = int(cfg.epochs * cfg.MULTIGRID.epoch_factor)
    for epoch in range(total_epochs):
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, epoch)
            if changed:
                logger.info("====== Rebuild model/optimizer/loader =====")
                (
                    model,
                    lr,
                    optimizer,
                    train_loader,
                    valid_loader,
                    precise_bn_loader,
                ) = build_trainer(cfg, places, parallel, validate, precise_bn,
                                  num_iters_precise_bn, world_size)

                #load checkpoint after re-build model
                if epoch != 0:
                    #epoch no need to -1, haved add 1 when save
                    filename = osp.join(
                        output_dir,
                        model_name + str(local_rank) + '_' + f"{(epoch):05d}")
                    subn_load(model, filename, optimizer)
                #update lr last epoch, not to use saved params
                lr.last_epoch = epoch
                lr.step(rebuild=True)

        model.train()
        record_list = build_record(cfg.MODEL)
        tic = time.time()
        for i, data in enumerate(train_loader):
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
            record_list['lr'].update(
                optimizer._global_learning_rate().numpy()[0], batch_size)
            for name, value in outputs.items():
                record_list[name].update(value.numpy()[0], batch_size)
            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, total_epochs, "train", ips)

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

        def evaluate(best):
            model.eval()
            record_list = build_record(cfg.MODEL)
            record_list.pop('lr')
            tic = time.time()
            for i, data in enumerate(valid_loader):
                outputs = model(data, mode='valid')

                # log_record
                for name, value in outputs.items():
                    record_list[name].update(value.numpy()[0], batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, total_epochs, "val",
                              ips)

            ips = "ips: {:.5f} instance/sec.".format(
                batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            if record_list.get('top1') and record_list['top1'].avg > best:
                best = record_list['top1'].avg
                best_flag = True
            return best, best_flag

        # use precise bn to improve acc
        if is_eval_epoch(cfg, epoch, total_epochs, multigrid.schedule):
            logger.info(f"do precise BN in {epoch+1} ...")
            do_preciseBN(model, precise_bn_loader, parallel,
                         min(num_iters_precise_bn, len(precise_bn_loader)))

        #  aggregate sub_BN stats
        logger.info("Aggregate sub_BatchNorm stats...")
        aggregate_sub_bn_stats(model)

        # 5. Validation
        if is_eval_epoch(cfg, epoch, total_epochs, multigrid.schedule):
            logger.info(f"eval in {epoch+1} ...")
            with paddle.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
                logger.info(
                    f"Already save the best model (top1 acc){int(best * 10000) / 10000}"
                )

        # 6. Save model and optimizer
        if is_eval_epoch(
                cfg, epoch,
                total_epochs, multigrid.schedule) or epoch % cfg.get(
                    "save_interval", 10) == 0 or epoch in multi_save_epoch:
            logger.info("[Save parameters] ======")
            subn_save(output_dir, model_name + str(local_rank) + '_', epoch + 1,
                      model, optimizer)

    logger.info(f'training {model_name} finished')
