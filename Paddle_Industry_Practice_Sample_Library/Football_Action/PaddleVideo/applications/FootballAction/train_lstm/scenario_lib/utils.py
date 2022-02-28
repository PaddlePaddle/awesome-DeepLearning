#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import time
import shutil
import numpy as np
import paddle.fluid as fluid
import logging

logger = logging.getLogger('LSTM')
best_test_acc1 = 0
min_test_loss = float("inf")


def log_lr_and_step():
    """log_lr_and_step"""
    try:
        # In optimizers, if learning_rate is set as constant, lr_var
        # name is 'learning_rate_0', and iteration counter is not
        # recorded. If learning_rate is set as decayed values from
        # learning_rate_scheduler, lr_var name is 'learning_rate',
        # and iteration counter is recorded with name '@LR_DECAY_COUNTER@',
        # better impliment is required here
        lr_var = fluid.global_scope().find_var("learning_rate")
        if not lr_var:
            lr_var = fluid.global_scope().find_var("learning_rate_0")
        lr = np.array(lr_var.get_tensor())

        lr_count = '[-]'
        lr_count_var = fluid.global_scope().find_var("@LR_DECAY_COUNTER@")
        if lr_count_var:
            lr_count = np.array(lr_count_var.get_tensor())
        # logger.info("------- learning rate {}, learning rate counter {} -----"
        #             .format(np.array(lr), np.array(lr_count)))
    except BaseException:
        logger.warn("Unable to get learning_rate and LR_DECAY_COUNTER.")


def test_with_pyreader(exe,
                       compiled_test_prog,
                       test_pyreader,
                       test_fetch_list,
                       test_metrics,
                       epoch,
                       log_interval=0,
                       save_model_name=''):
    """test_with_pyreader"""
    if not test_pyreader:
        logger.error("[TEST] get pyreader failed.")

    test_loss = -1
    test_acc1 = 0
    test_status = False
    for retry in range(3):
        test_metrics.reset()
        test_iter = 0

        try:
            for data in test_pyreader():
                test_outs = exe.run(compiled_test_prog,
                                    fetch_list=test_fetch_list,
                                    feed=data)
                loss = np.array(test_outs[0])
                pred_label = np.array(test_outs[1])
                pred_iou = np.array(test_outs[2])
                label = np.array(test_outs[-2])
                iou = np.array(test_outs[-1])

                test_metrics.accumulate(loss, pred_label, label, pred_iou, iou)
                test_iter += 1
            test_loss, test_acc1, test_iou = test_metrics.finalize_and_log_out( \
                info='[TEST] Finish Epoch {}: '.format(epoch))
            test_status = True
            return test_status, test_loss, test_acc1, test_iou
        except Exception as e:
            logger.warn(
                "[TEST] Epoch {} fail to execute test or calculate metrics: {}".
                format(epoch, e))
    logger.warn(
        "[TEST] Finish...  Epoch {} fail to execute test or calculate metrics.".
        format(epoch))
    return test_status, test_loss, test_acc1, test_iou


def train_with_pyreader(exe, train_prog, compiled_train_prog, train_pyreader, \
                        train_fetch_list, train_metrics, epochs=10, \
                        log_interval=0, valid_interval=0, save_dir='./', \
                        save_model_name='model', \
                        compiled_test_prog=None, test_pyreader=None, \
                        test_fetch_list=None, test_metrics=None):
    """train_with_pyreader"""
    if not train_pyreader:
        logger.error("[TRAIN] get pyreader failed.")
    epoch_periods = []
    train_loss = 0
    for epoch in range(epochs):

        train_metrics.reset()
        train_iter = 0
        epoch_periods = []

        for data in train_pyreader():
            # logger.info("epoch = {} train_iter = {}".format(epoch, train_iter))
            try:
                cur_time = time.time()
                train_outs = exe.run(compiled_train_prog,
                                     fetch_list=train_fetch_list,
                                     feed=data)
                log_lr_and_step()
                period = time.time() - cur_time
                epoch_periods.append(period)
                loss = np.array(train_outs[0])
                pred_label = np.array(train_outs[1])
                pred_iou = np.array(train_outs[2])
                label = np.array(train_outs[-2])
                iou = np.array(train_outs[-1])
                train_metrics.accumulate(loss, pred_label, label, pred_iou, iou)
                if log_interval > 0 and (train_iter % log_interval == 0):
                    train_metrics.finalize_and_log_out( \
                        info='[TRAIN] Epoch {}, iter {} average: '.format(epoch, train_iter))
            except Exception as e:
                logger.info(
                    "[TRAIN] Epoch {}, iter {} data training failed: {}".format(
                        epoch, train_iter, str(e)))
            train_iter += 1

        if len(epoch_periods) < 1:
            logger.info(
                'No iteration was executed, please check the data reader')
            sys.exit(1)

        logger.info(
            '[TRAIN] Epoch {} training finished, average time: {}'.format(
                epoch, np.mean(epoch_periods)))
        train_metrics.finalize_and_log_out( \
            info='[TRAIN] Finished ... Epoch {} all iters average: '.format(epoch))

        #save_postfix = "_epoch{}".format(epoch)
        #save_model(exe, train_prog, save_dir, save_model_name, save_postfix)

        # save models of min loss in best acc epochs
        if compiled_test_prog and valid_interval > 0 and (
                epoch + 1) % valid_interval == 0:
            test_status, test_loss, test_acc1, test_iou = test_with_pyreader(
                exe, compiled_test_prog, test_pyreader, test_fetch_list,
                test_metrics, epoch, log_interval, save_model_name)
            global best_test_acc1
            global min_test_loss
            if test_status and (test_acc1 > best_test_acc1 or
                                (test_acc1 == best_test_acc1
                                 and test_loss < min_test_loss)):
                best_test_acc1 = test_acc1
                min_test_loss = test_loss
                save_postfix = "_epoch{}_acc{}".format(epoch, best_test_acc1)
                save_model(exe, train_prog, save_dir, save_model_name,
                           save_postfix)


def save_model(exe, program, save_dir, model_name, postfix=None):
    """save_model"""
    #model_path = os.path.join(save_dir, model_name + postfix)
    #if os.path.isdir(model_path):
    #    shutil.rmtree(model_path)
    ##fluid.io.save_persistables(exe, model_path, main_program=program)
    #save_vars = [x for x in program.list_vars() \
    #             if isinstance(x, fluid.framework.Parameter)]
    #fluid.io.save_vars(exe, dirname=model_path, main_program=program, vars=save_vars, filename="param")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    saved_model_name = model_name + postfix

    fluid.save(program, os.path.join(save_dir, saved_model_name))

    return
