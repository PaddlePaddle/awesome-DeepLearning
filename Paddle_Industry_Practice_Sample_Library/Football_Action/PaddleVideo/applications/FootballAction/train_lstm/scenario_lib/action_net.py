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
import math
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import ParamAttr

import logging

logger = logging.getLogger('LSTM')

def is_parameter(var):
    """is_parameter"""
    return isinstance(var, fluid.framework.Parameter)


class ActionNet:
    """ActionNet"""

    def __init__(self, name, cfg, mode='train'):
        self.cfg = cfg
        self.name = name
        self.mode = mode
        self.py_reader = None
        self.get_config()

    def get_config(self):
        """get_config"""
        # get model configs
        self.feature_num = self.cfg.MODEL.feature_num
        self.feature_names = self.cfg.MODEL.feature_names
        self.with_bn = self.cfg.MODEL.with_bn
        self.feature_dims = self.cfg.MODEL.feature_dims
        self.num_classes = self.cfg.MODEL.num_classes
        self.embedding_size = self.cfg.MODEL.embedding_size
        #self.lstm_size = self.cfg.MODEL.lstm_size
        self.lstm_size_img = self.cfg.MODEL.lstm_size_img
        self.lstm_size_audio = self.cfg.MODEL.lstm_size_audio
        self.drop_rate = self.cfg.MODEL.drop_rate
        self.save_dir = self.cfg.MODEL.save_dir

        # get mode configs
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size', 1)
        self.num_gpus = self.get_config_from_sec(self.mode, 'num_gpus', 1)

        if self.mode == 'train':
            self.learning_rate = self.get_config_from_sec('train',
                                                          'learning_rate', 1e-3)
            self.weight_decay = self.get_config_from_sec('train',
                                                         'weight_decay', 8e-4)
            self.num_samples = self.get_config_from_sec('train', 'num_samples',
                                                        5000000)
            self.decay_epochs = self.get_config_from_sec('train',
                                                         'decay_epochs', [5])
            self.decay_gamma = self.get_config_from_sec('train', 'decay_gamma',
                                                        0.1)
            self.droplast = self.get_config_from_sec('train', 'droplast', False)

    def get_config_from_sec(self, sec, item, default=None):
        """get_config_from_sec"""
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    def pyreader(self):
        """pyreader"""
        return self.py_reader

    def load_pretrain_params_file(self, exe, pretrain, prog, place):
        logger.info("Load pretrain weights from {}, param".format(pretrain))

        load_vars = [x for x in prog.list_vars() \
                         if isinstance(x, fluid.framework.Parameter) and x.name.find('fc_8') == -1]
        fluid.io.load_vars(exe, dirname=pretrain, vars=load_vars, filename="param")

    def load_test_weights_file(self, exe, weights, prog, place):
        params_list = list(filter(is_parameter, prog.list_vars()))
        fluid.load(prog, weights, executor=exe, var_list=params_list)
    #def load_test_weights_file(self, exe, weights, prog, place):
    #    """load_test_weights_file"""
    #    load_vars = [x for x in prog.list_vars() \
    #                 if isinstance(x, fluid.framework.Parameter)]
    #    fluid.io.load_vars(exe, dirname=weights, vars=load_vars, filename="param")

    def epoch_num(self):
        """get train epoch num"""
        return self.cfg.TRAIN.epoch


    def build_input(self, use_pyreader):
        """build_input"""
        self.feature_input = []
        for name, dim in zip(self.feature_names, self.feature_dims):
            self.feature_input.append(
                fluid.layers.data(
                    shape=[dim], lod_level=1, dtype='float32', name=name))
        if self.mode != 'infer':
            self.label_id_input = fluid.layers.data(
                shape=[1], dtype='int64', name='label_cls')
            self.label_iou_input = fluid.layers.data(
                shape=[1], dtype='float32', name='label_iou')
        else:
            self.label_id_input = None
            self.label_iou_input = None
        if use_pyreader:
            assert self.mode != 'infer', \
                'pyreader is not recommendated when infer, please set use_pyreader to be false.'
            self.py_reader = fluid.io.PyReader(
                feed_list=self.feature_input + [self.label_id_input] + [self.label_iou_input],
                capacity=1024,
                iterable=True)


    def build_model(self):
        """build_model"""
        # ---------------- transfer from old paddle ---------------
        # ------image------

        lstm_forward_fc = fluid.layers.fc(
            input=self.feature_input[0],
            size=self.lstm_size_img * 4,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        lstm_forward, _ = fluid.layers.dynamic_lstm(input=lstm_forward_fc, size=self.lstm_size_img * 4, is_reverse=False,
                                                    use_peepholes=True)
        #lstm_forward_add = fluid.layers.elementwise_add(self.feature_input[0], lstm_forward, act='relu')
        #print("lstm_backward_add.shape", lstm_forward_add.shape)

        lstm_backward_fc = fluid.layers.fc(
            input=self.feature_input[0],
            size=self.lstm_size_img * 4,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        lstm_backward, _ = fluid.layers.dynamic_lstm(input=lstm_backward_fc, size=self.lstm_size_img * 4, is_reverse=True,
                                                     use_peepholes=True)
        #lstm_backward_add = fluid.layers.elementwise_add(self.feature_input[0], lstm_backward, act='relu')
        #print("lstm_backward_add.shape", lstm_backward_add.shape)

        #lstm_img = fluid.layers.concat(input=[lstm_forward_add, lstm_backward_add], axis=1)
        lstm_img = fluid.layers.concat(input=[lstm_forward, lstm_backward], axis=1)
        print("lstm_img.shape", lstm_img.shape)

        lstm_dropout = fluid.layers.dropout(x=lstm_img, dropout_prob=self.drop_rate,
                                            is_test=(not self.mode == 'train'))
        lstm_weight = fluid.layers.fc(
            input=lstm_dropout,
            size=1,
            act='sequence_softmax',
            bias_attr=None)

        scaled = fluid.layers.elementwise_mul(x=lstm_dropout, y=lstm_weight, axis=0)
        lstm_pool = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        # ------audio------
        lstm_forward_fc_audio = fluid.layers.fc(
            input=self.feature_input[1],
            size=self.lstm_size_audio * 4,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        lstm_forward_audio, _ = fluid.layers.dynamic_lstm(
            input=lstm_forward_fc_audio, size=self.lstm_size_audio * 4, is_reverse=False, use_peepholes=True)

        lsmt_backward_fc_audio = fluid.layers.fc(
            input=self.feature_input[1],
            size=self.lstm_size_audio * 4,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        lstm_backward_audio, _ = fluid.layers.dynamic_lstm(input=lsmt_backward_fc_audio, size=self.lstm_size_audio * 4,
                                                           is_reverse=True, use_peepholes=True)

        lstm_forward_audio = fluid.layers.concat(input=[lstm_forward_audio, lstm_backward_audio], axis=1)

        lstm_dropout_audio = fluid.layers.dropout(x=lstm_forward_audio, dropout_prob=self.drop_rate,
                                                  is_test=(not self.mode == 'train'))
        lstm_weight_audio = fluid.layers.fc(
            input=lstm_dropout_audio,
            size=1,
            act='sequence_softmax',
            bias_attr=None)

        scaled_audio = fluid.layers.elementwise_mul(x=lstm_dropout_audio, y=lstm_weight_audio, axis=0)
        lstm_pool_audio = fluid.layers.sequence_pool(input=scaled_audio, pool_type='sum')
        # ------ concat -------
        lstm_concat = fluid.layers.concat(input=[lstm_pool, lstm_pool_audio], axis=1)
        #print("lstm_concat.shape", lstm_concat.shape)

        input_fc_proj = fluid.layers.fc(
            input=lstm_concat,
            # input=lstm_pool,      # 只用image feature训练
            size=8192,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        input_fc_proj_bn = fluid.layers.batch_norm(input=input_fc_proj, act="relu",
            is_test=(not self.mode == 'train'))
        # model remove bn when batch_size is small
        if not self.with_bn:
            input_fc_proj_bn = 0 * input_fc_proj_bn + input_fc_proj
        input_fc_proj_dropout = fluid.layers.dropout(x=input_fc_proj_bn, dropout_prob=self.drop_rate,
            is_test=(not self.mode == 'train'))

        input_fc_hidden = fluid.layers.fc(
            input=input_fc_proj_dropout,
            size=4096,
            act=None,
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)))
        input_fc_hidden_bn = fluid.layers.batch_norm(input=input_fc_hidden, act="relu",
            is_test=(not self.mode == 'train'))
        # model remove bn when batch_size is small
        if not self.with_bn:
            input_fc_hidden_bn = 0 * input_fc_hidden_bn + input_fc_hidden
        input_fc_hidden_dropout = fluid.layers.dropout(x=input_fc_hidden_bn, dropout_prob=self.drop_rate,
            is_test=(not self.mode == 'train'))
        self.fc = fluid.layers.fc(
            input=input_fc_hidden_dropout,
            size=self.num_classes,
            act='softmax')
        self.fc_iou = fluid.layers.fc(
            input=input_fc_hidden_dropout,
            size=1,
            act="sigmoid")
        self.network_outputs = [self.fc, self.fc_iou]
        
    def optimizer(self):
        """optimizer"""
        assert self.mode == 'train', "optimizer only can be get in train mode"
        values = [
            self.learning_rate * (self.decay_gamma**i)
            for i in range(len(self.decay_epochs) + 1)
        ]
        if self.droplast:
            self.num_samples = math.floor(float(self.num_samples) / float(self.batch_size)) * self.batch_size
            iter_per_epoch = math.floor(float(self.num_samples) / self.batch_size)
        else:
            self.num_samples = math.ceil(float(self.num_samples) / float(self.batch_size)) * self.batch_size
            iter_per_epoch = math.ceil(float(self.num_samples) / self.batch_size)

        boundaries = [e * iter_per_epoch for e in self.decay_epochs]
        logger.info("num_sample = {}, batchsize = {}, iter_per_epoch = {}, lr_int = {}, boundaries = {} "
                    .format(self.num_samples, self.batch_size, \
                            iter_per_epoch, self.learning_rate, np.array(boundaries)))

        return fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.piecewise_decay(
                values=values, boundaries=boundaries),
            centered=True,
            regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))

    def _calc_label_smoothing_loss(self, softmax_out, label, class_dim, epsilon=0.1):
        """Calculate label smoothing loss
           Returns:
           label smoothing loss
        """
        label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
        smooth_label = fluid.layers.label_smooth(
            label=label_one_hot, epsilon=epsilon, dtype="float32")
        loss = fluid.layers.cross_entropy(
            input=softmax_out, label=smooth_label, soft_label=True)
        return loss

    def loss(self):
        """
        loss
        """
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost_cls = fluid.layers.cross_entropy(input=self.network_outputs[0], label=self.label_id_input)
        cost_cls = fluid.layers.reduce_sum(cost_cls, dim=-1)
        sum_cost_cls = fluid.layers.reduce_sum(cost_cls)
        self.loss_cls_ = fluid.layers.scale(sum_cost_cls, scale=self.num_gpus, bias_after_scale=False)
        cost_iou = fluid.layers.square_error_cost(input=self.network_outputs[1], label=self.label_iou_input)
        cost_iou = fluid.layers.reduce_sum(cost_iou, dim=-1)
        sum_cost_iou = fluid.layers.reduce_sum(cost_iou)
        self.loss_iou_ = fluid.layers.scale(sum_cost_iou, scale=self.num_gpus, bias_after_scale=False)
        alpha = 10
        self.loss_ = self.loss_cls_ + alpha * self.loss_iou_
        return self.loss_


    def outputs(self):
        """outputs"""
        return self.network_outputs

    def feeds(self):
        """
        feeds
        """
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_id_input, self.label_iou_input]

    def fetches(self):
        """fetches"""
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.network_outputs[1], \
                          self.label_id_input, self.label_iou_input]
        elif self.mode == 'infer':
            fetch_list = [self.network_outputs[0], self.network_outputs[1]]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list
