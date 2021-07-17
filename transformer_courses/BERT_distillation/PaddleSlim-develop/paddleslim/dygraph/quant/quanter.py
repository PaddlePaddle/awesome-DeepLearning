# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import copy
import logging

import paddle
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from ...common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

WEIGHT_QUANTIZATION_TYPES = [
    'abs_max', 'channel_wise_abs_max', 'range_abs_max', 'moving_average_abs_max'
]

ACTIVATION_QUANTIZATION_TYPES = [
    'abs_max', 'range_abs_max', 'moving_average_abs_max'
]

BUILT_IN_PREPROCESS_TYPES = ['PACT']

VALID_DTYPES = ['int8']

__all__ = ['QAT']

_quant_config_default = {
    # weight preprocess type, default is None and no preprocessing is performed. 
    'weight_preprocess_type': None,
    # activation preprocess type, default is None and no preprocessing is performed.
    'activation_preprocess_type': None,
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. default is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}


def _parse_configs(user_config):
    """
    check if user's configs are valid.
    Args:
        user_config(dict): user's config.
    Return:
        configs(dict): final configs will be used.
    """

    configs = copy.deepcopy(_quant_config_default)
    configs.update(user_config)

    # check if configs is valid
    weight_types = WEIGHT_QUANTIZATION_TYPES
    activation_types = WEIGHT_QUANTIZATION_TYPES

    assert configs['weight_preprocess_type'] in BUILT_IN_PREPROCESS_TYPES or configs['weight_preprocess_type'] is None, \
        "Unknown weight_preprocess_type: {}. only supports {} ".format(configs['weight_preprocess_type'],
                BUILT_IN_PREPROCESS_TYPES)

    assert configs['activation_preprocess_type'] in BUILT_IN_PREPROCESS_TYPES or configs['activation_preprocess_type'] is None, \
        "Unknown activation_preprocess_type: {}. only supports {}".format(configs['activation_preprocess_type'],
                BUILT_IN_PREPROCESS_TYPES)

    assert configs['weight_quantize_type'] in WEIGHT_QUANTIZATION_TYPES, \
        "Unknown weight_quantize_type: {}. only supports {} ".format(configs['weight_quantize_type'],
                WEIGHT_QUANTIZATION_TYPES)

    assert configs['activation_quantize_type'] in ACTIVATION_QUANTIZATION_TYPES, \
        "Unknown activation_quantize_type: {}. only supports {}".format(configs['activation_quantize_type'],
                ACTIVATION_QUANTIZATION_TYPES)

    assert isinstance(configs['weight_bits'], int), \
        "weight_bits must be int value."

    assert (configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16), \
        "weight_bits should be between 1 and 16."

    assert isinstance(configs['activation_bits'], int), \
        "activation_bits must be int value."

    assert (configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16), \
        "activation_bits should be between 1 and 16."

    assert isinstance(configs['dtype'], str), \
        "dtype must be a str."

    assert (configs['dtype'] in VALID_DTYPES), \
        "dtype can only be " + " ".join(VALID_DTYPES)

    assert isinstance(configs['window_size'], int), \
        "window_size must be int value, window size for 'range_abs_max' quantization, default is 10000."

    assert isinstance(configs['moving_rate'], float), \
        "moving_rate must be float value, The decay coefficient of moving average, default is 0.9."

    assert isinstance(configs['quantizable_layer_type'], list), \
        "quantizable_layer_type must be a list"

    return configs


class PACT(paddle.nn.Layer):
    def __init__(self):
        super(PACT, self).__init__()
        alpha_attr = paddle.ParamAttr(
            name=self.full_name() + ".pact",
            initializer=paddle.nn.initializer.Constant(value=20),
            learning_rate=1000.0)

        self.alpha = self.create_parameter(
            shape=[1], attr=alpha_attr, dtype='float32')

    def forward(self, x):
        out_left = paddle.nn.functional.relu(x - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        x = x - out_left + out_right
        return x


class QAT(object):
    """
    Quant Aware Training(QAT): Add the fake quant logic for given quantizable layers, namely add the quant_dequant computational logic both for activation inputs and weight inputs.
    """

    def __init__(self,
                 config=None,
                 weight_preprocess=None,
                 act_preprocess=None,
                 weight_quantize=None,
                 act_quantize=None):
        """
        Args:
            model(nn.Layer)
            config(dict, optional): configs for quantization. if None, will use default config. 
                    Default: None.
            weight_quantize(class, optional): Defines how to quantize weight. Using this
                    can quickly test if user's quantization method works or not. In this method, user should
                    both define quantization function and dequantization function, that is, the function's input
                    is non-quantized weight and function returns dequantized weight. If None, will use
                    quantization op defined by 'weight_quantize_type'.
                    Default is None.
            act_quantize(class, optional): Defines how to quantize activation. Using this
                    can quickly test if user's quantization method works or not. In this function, user should
                    both define quantization and dequantization process, that is, the function's input
                    is non-quantized activation and function returns dequantized activation. If None, will use 
                    quantization op defined by 'activation_quantize_type'.
                    Default is None.
            weight_preprocess(class, optional): Defines how to preprocess weight before quantization. Using this
                    can quickly test if user's preprocess method works or not. The function's input
                    is non-quantized weight and function returns processed weight to be quantized. If None, will
                    use preprocess method defined by 'weight_preprocess_type'.
                    Default is None.
            act_preprocess(class, optional): Defines how to preprocess activation before quantization. Using this
                    can quickly test if user's preprocess method works or not. The function's input
                    is non-quantized activation and function returns processed activation to be quantized. If None,
                    will use preprocess method defined by 'activation_preprocess_type'.
                    Default is None.
        """
        if config is None:
            config = _quant_config_default
        else:
            assert isinstance(config, dict), "config must be dict"
            config = _parse_configs(config)
        self.config = config

        self.weight_preprocess = PACT if self.config[
            'weight_preprocess_type'] == 'PACT' else None
        self.act_preprocess = PACT if self.config[
            'activation_preprocess_type'] == 'PACT' else None

        self.weight_preprocess = weight_preprocess if weight_preprocess is not None else self.weight_preprocess
        self.act_preprocess = act_preprocess if act_preprocess is not None else self.act_preprocess
        self.weight_quantize = weight_quantize
        self.act_quantize = act_quantize

        self.imperative_qat = ImperativeQuantAware(
            weight_bits=self.config['weight_bits'],
            activation_bits=self.config['activation_bits'],
            weight_quantize_type=self.config['weight_quantize_type'],
            activation_quantize_type=self.config['activation_quantize_type'],
            moving_rate=self.config['moving_rate'],
            quantizable_layer_type=self.config['quantizable_layer_type'],
            weight_preprocess_layer=self.weight_preprocess,
            act_preprocess_layer=self.act_preprocess,
            weight_quantize_layer=self.weight_quantize,
            act_quantize_layer=self.act_quantize)

    def quantize(self, model):
        self._model = copy.deepcopy(model)
        self.imperative_qat.quantize(model)

    def save_quantized_model(self, model, path, input_spec=None):
        if self.weight_preprocess is not None or self.act_preprocess is not None:
            training = model.training
            model = self._remove_preprocess(model)
            if training:
                model.train()
            else:
                model.eval()

        self.imperative_qat.save_quantized_model(
            layer=model, path=path, input_spec=input_spec)

    def _remove_preprocess(self, model):
        state_dict = model.state_dict()
        self.imperative_qat = ImperativeQuantAware(
            weight_bits=self.config['weight_bits'],
            activation_bits=self.config['activation_bits'],
            weight_quantize_type=self.config['weight_quantize_type'],
            activation_quantize_type=self.config['activation_quantize_type'],
            moving_rate=self.config['moving_rate'],
            quantizable_layer_type=self.config['quantizable_layer_type'])

        with paddle.utils.unique_name.guard():
            if hasattr(model, "_layers"):
                model = model._layers
            model = self._model
            self.imperative_qat.quantize(model)
            model.set_state_dict(state_dict)

        return model
