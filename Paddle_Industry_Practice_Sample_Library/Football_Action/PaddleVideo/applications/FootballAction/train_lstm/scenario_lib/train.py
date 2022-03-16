#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import os
import sys
import time
import argparse
import ast
import six
import numpy as np
import paddle.fluid as fluid
import accuracy_metrics
import feature_reader
import config
import action_net
import utils
import logging
import paddle

paddle.enable_static()
o_path = os.getcwd()
sys.path.append(o_path)
#logger = loginfo.Logger()
logger = logging.getLogger('LSTM')
logger.setLevel(logging.INFO)


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument('--model_name',
                        type=str,
                        default='BaiduNet',
                        help='name of model to train.')
    parser.add_argument('--config',
                        type=str,
                        default='configs/conf.txt',
                        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help=
        'path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument('--use_gpu',
                        type=ast.literal_eval,
                        default=True,
                        help='default use gpu.')
    parser.add_argument('--no_memory_optimize',
                        action='store_true',
                        default=False,
                        help='whether to use memory optimize in train')
    parser.add_argument('--epoch_num',
                        type=int,
                        default=0,
                        help='epoch number, 0 for read from config file')
    parser.add_argument('--valid_interval',
                        type=int,
                        default=1,
                        help='validation epoch interval, 0 for no validation.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='checkpoints',
                        help='directory name to save train snapshoot')
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def print_prog(prog):
    """print_prog"""
    for name, value in sorted(six.iteritems(prog.block(0).vars)):
        logger.info(value)
    for op in prog.block(0).ops:
        logger.info("op type is {}".format(op.type))
        logger.info("op inputs are {}".format(op.input_arg_names))
        logger.info("op outputs are {}".format(op.output_arg_names))
        for key, value in sorted(six.iteritems(op.all_attrs())):
            if key not in ['op_callstack', 'op_role_var']:
                logger.info(" [ attrs: {}:   {} ]".format(key, value))


def train(args):
    """train"""
    logger.info("Start train program")
    # parse config
    config_info = config.parse_config(args.config)
    train_config = config.merge_configs(config_info, 'train', vars(args))
    valid_config = config.merge_configs(config_info, 'valid', vars(args))
    valid_config['MODEL']['save_dir'] = args.save_dir

    bs_denominator = 1
    if args.use_gpu:
        # check number of GPUs
        gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if gpus == "":
            pass
        else:
            gpus = gpus.split(",")
            num_gpus = len(gpus)
            assert num_gpus == train_config.TRAIN.num_gpus, \
                "num_gpus({}) set by CUDA_VISIBLE_DEVICES" \
                "shoud be the same as that" \
                "set in {}({})".format(
                    num_gpus, args.config, train_config.TRAIN.num_gpus)
        bs_denominator = train_config.TRAIN.num_gpus

    # adaptive batch size
    train_batch_size_in = train_config.TRAIN.batch_size
    #  train_learning_rate_in = train_config.TRAIN.learning_rate
    train_config.TRAIN.batch_size = min(
        int(train_config.TRAIN.num_samples / 10), train_batch_size_in)
    train_config.TRAIN.batch_size = int(
        train_config.TRAIN.batch_size / bs_denominator) * bs_denominator
    train_config.TRAIN.batch_size = max(train_config.TRAIN.batch_size,
                                        bs_denominator)
    # train_config.TRAIN.learning_rate = float(train_learning_rate_in) / float(train_batch_size_in) \
    #     * train_config.TRAIN.batch_size

    val_batch_size_in = valid_config.VALID.batch_size
    valid_config.VALID.batch_size = min(
        int(valid_config.VALID.num_samples / 10), val_batch_size_in)
    valid_config.VALID.batch_size = int(
        valid_config.VALID.batch_size / bs_denominator) * bs_denominator
    valid_config.VALID.batch_size = max(valid_config.VALID.batch_size,
                                        bs_denominator)

    # model remove bn when train every gpu batch_size is small
    if int(train_config.TRAIN.batch_size /
           bs_denominator) < train_config.MODEL.modelbn_min_everygpu_bs:
        train_config.MODEL.with_bn = False
        valid_config.MODEL.with_bn = False
    else:
        train_config.MODEL.with_bn = True
        valid_config.MODEL.with_bn = True

    config.print_configs(train_config, 'Train')
    train_model = action_net.ActionNet(args.model_name,
                                       train_config,
                                       mode='train')
    valid_model = action_net.ActionNet(args.model_name,
                                       valid_config,
                                       mode='valid')

    # build model
    startup = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup):
        with fluid.unique_name.guard():
            train_model.build_input(use_pyreader=True)
            train_model.build_model()
            # for the input, has the form [data1, data2,..., label], so train_feeds[-1] is label
            train_feeds = train_model.feeds()
            train_fetch_list = train_model.fetches()
            train_loss = train_fetch_list[0]
            for item in train_fetch_list:
                item.persistable = True
            optimizer = train_model.optimizer()
            optimizer.minimize(train_loss)
            train_pyreader = train_model.pyreader()

    valid_prog = fluid.Program()
    with fluid.program_guard(valid_prog, startup):
        with fluid.unique_name.guard():
            valid_model.build_input(use_pyreader=True)
            valid_model.build_model()
            valid_feeds = valid_model.feeds()
            valid_fetch_list = valid_model.fetches()
            valid_pyreader = valid_model.pyreader()
            for item in valid_fetch_list:
                item.persistable = True

    valid_prog = valid_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    #print_prog(train_prog)
    #print_prog(valid_prog)

    if args.resume:
        # if resume weights is given, load resume weights directly
        assert os.path.exists(args.resume), \
            "Given resume weight dir {} not exist.".format(args.resume)

        def if_exist(var):
            return os.path.exists(os.path.join(args.resume, var.name))

        fluid.io.load_vars(exe,
                           args.resume,
                           predicate=if_exist,
                           main_program=train_prog)
    else:
        # if not in resume mode, load pretrain weights
        if args.pretrain:
            assert os.path.exists(args.pretrain), \
                "Given pretrain weight dir {} not exist.".format(args.pretrain)
            pretrain = args.pretrain or train_model.get_pretrain_weights()
            if pretrain:
                train_model.load_pretrain_params_file(exe, pretrain, train_prog,
                                                      place)

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True

    compiled_train_prog = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(loss_name=train_loss.name,
                                       build_strategy=build_strategy)
    compiled_valid_prog = fluid.compiler.CompiledProgram(
        valid_prog).with_data_parallel(share_vars_from=compiled_train_prog,
                                       build_strategy=build_strategy)
    # get reader
    train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size /
                                        bs_denominator)
    valid_config.VALID.batch_size = int(valid_config.VALID.batch_size /
                                        bs_denominator)
    print("config setting")
    train_dataload = feature_reader.FeatureReader(args.model_name.upper(),
                                                  'train', train_config,
                                                  bs_denominator)
    train_reader = train_dataload.create_reader()
    print("train reader")
    valid_dataload = feature_reader.FeatureReader(args.model_name.upper(),
                                                  'valid', valid_config,
                                                  bs_denominator)
    valid_reader = valid_dataload.create_reader()

    # get metrics
    train_metrics = accuracy_metrics.MetricsCalculator(args.model_name.upper(),
                                                       'train', train_config)
    valid_metrics = accuracy_metrics.MetricsCalculator(args.model_name.upper(),
                                                       'valid', valid_config)

    epochs = args.epoch_num or train_model.epoch_num()
    print("epoch is ", epochs)

    exe_places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    train_pyreader.decorate_sample_list_generator(train_reader,
                                                  places=exe_places)
    valid_pyreader.decorate_sample_list_generator(valid_reader,
                                                  places=exe_places)

    utils.train_with_pyreader(
        exe,
        train_prog,
        compiled_train_prog,  # train_exe,
        train_pyreader,
        train_fetch_list,
        train_metrics,
        epochs=epochs,
        log_interval=args.log_interval,
        valid_interval=args.valid_interval,
        save_dir=args.save_dir,
        save_model_name=args.model_name,
        compiled_test_prog=compiled_valid_prog,  # test_exe=valid_exe,
        test_pyreader=valid_pyreader,
        test_fetch_list=valid_fetch_list,
        test_metrics=valid_metrics)

    logger.info("Finish program")


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
