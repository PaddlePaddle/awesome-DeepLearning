# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy
import json
import logging

import paddle
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import ConvertToInt8Pass
from paddle.fluid.contrib.slim.quantization import TransformForMobilePass
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPass
from paddle.fluid.contrib.slim.quantization import OutScaleForTrainingPass
from paddle.fluid.contrib.slim.quantization import OutScaleForInferencePass
from paddle.fluid import core
from paddle.fluid.contrib.slim.quantization import WeightQuantization

from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)

WEIGHT_QUANTIZATION_TYPES = [
    'abs_max', 'channel_wise_abs_max', 'range_abs_max', 'moving_average_abs_max'
]
WEIGHT_QUANTIZATION_TYPES_TENSORRT = ['channel_wise_abs_max']

ACTIVATION_QUANTIZATION_TYPES = [
    'abs_max', 'range_abs_max', 'moving_average_abs_max'
]

ACTIVATION_QUANTIZATION_TYPES_TENSORRT = [
    'range_abs_max', 'moving_average_abs_max'
]

VALID_DTYPES = ['int8']
TRANSFORM_PASS_OP_TYPES = QuantizationTransformPass._supported_quantizable_op_type
QUANT_DEQUANT_PASS_OP_TYPES = AddQuantDequantPass._supported_quantizable_op_type

TENSORRT_OP_TYPES = [
    'mul', 'conv2d', 'pool2d', 'depthwise_conv2d', 'elementwise_add',
    'leaky_relu'
]

VARS_MAPPING_TABLE = './mapping_table_for_saving_inference_model'

_quant_config_default = {
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # ops of name_scope in not_quant_pattern list, will not be quantized
    'not_quant_pattern': ['skip_quant'],
    # ops of type in quantize_op_types, will be quantized
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. defaulf is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # if True, 'quantize_op_types' will be TENSORRT_OP_TYPES
    'for_tensorrt': False,
    # if True, 'quantoze_op_types' will be TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES 
    'is_full_quantize': False
}


def load_dict():
    with open(VARS_MAPPING_TABLE, 'r') as file:
        data = file.read()
        data = json.loads(data)
        return data


def save_dict(table):
    with open(VARS_MAPPING_TABLE, 'w') as file:
        file.write(json.dumps(table))


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

    assert isinstance(configs['for_tensorrt'], bool) and isinstance(
        configs['is_full_quantize'],
        bool), "'for_tensorrt' and 'is_full_quantize' must both be bool'"

    # check if configs is valid
    if configs['for_tensorrt']:
        weight_types = WEIGHT_QUANTIZATION_TYPES_TENSORRT
        activation_types = ACTIVATION_QUANTIZATION_TYPES_TENSORRT
        platform = 'TensorRT'
    else:
        weight_types = WEIGHT_QUANTIZATION_TYPES
        activation_types = WEIGHT_QUANTIZATION_TYPES
        platform = 'PaddleLite'
    assert configs['weight_quantize_type'] in weight_types, \
        "Unknown weight_quantize_type: {}. {} only supports {} ".format(configs['weight_quantize_type'],
                platform, weight_types)

    assert configs['activation_quantize_type'] in activation_types, \
        "Unknown activation_quantize_type: {}. {} only supports {}".format(configs['activation_quantize_type'],
                platform, activation_types)

    assert isinstance(configs['weight_bits'], int), \
        "weight_bits must be int value."

    assert (configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16), \
        "weight_bits should be between 1 and 16."

    assert isinstance(configs['activation_bits'], int), \
        "activation_bits must be int value."

    assert (configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16), \
        "activation_bits should be between 1 and 16."

    assert isinstance(configs['not_quant_pattern'], (list, str)), \
        "not_quant_pattern must be list or str"

    assert isinstance(configs['quantize_op_types'], list), \
        "quantize_op_types must be a list"

    if configs['for_tensorrt']:
        configs['quantize_op_types'] = TENSORRT_OP_TYPES
    elif configs['is_full_quantize']:
        configs[
            'quantize_op_types'] = TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
    else:
        for op_type in configs['quantize_op_types']:
            assert (op_type in QUANT_DEQUANT_PASS_OP_TYPES) or (
                op_type in TRANSFORM_PASS_OP_TYPES), "{} is not support, \
                        now support op types are {}".format(
                    op_type,
                    TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES)

    assert isinstance(configs['dtype'], str), \
        "dtype must be a str."

    assert (configs['dtype'] in VALID_DTYPES), \
        "dtype can only be " + " ".join(VALID_DTYPES)

    assert isinstance(configs['window_size'], int), \
        "window_size must be int value, window size for 'range_abs_max' quantization, default is 10000."

    assert isinstance(configs['moving_rate'], float), \
        "moving_rate must be float value, The decay coefficient of moving average, default is 0.9."

    return configs


def quant_aware(program,
                place,
                config=None,
                scope=None,
                for_test=False,
                weight_quantize_func=None,
                act_quantize_func=None,
                weight_preprocess_func=None,
                act_preprocess_func=None,
                optimizer_func=None,
                executor=None,
                return_program=False):
    """Add quantization  and dequantization operators to "program" 
    for quantization training or testing.

    Args:
        program(paddle.static.Program): training or testing ``program``.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents 
            the executor run on which device.
        config(dict, optional): configs for quantization. if None, will use default config. 
            Default: None.
        scope(paddle.static.Scope): Scope records the mapping between variable names and variables, 
            similar to brackets in programming languages. Usually users can use 
            `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.              When ``None`` will use `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_ . Default: ``None``.
        for_test(bool): If the 'program' parameter is a test program, this parameter should be set to ``True``. 
            Otherwise, set to ``False``.Default: False
       weight_quantize_func(function): Function that defines how to quantize weight. Using this
                can quickly test if user's quantization method works or not. In this function, user should
                both define quantization function and dequantization function, that is, the function's input
                is non-quantized weight and function returns dequantized weight. If None, will use
                quantization op defined by 'weight_quantize_type'.
                Default is None.
        act_quantize_func(function): Function that defines how to quantize activation. Using this
                can quickly test if user's quantization method works or not. In this function, user should
                both define quantization and dequantization process, that is, the function's input
                is non-quantized activation and function returns dequantized activation. If None, will use 
                quantization op defined by 'activation_quantize_type'.
                Default is None.
        weight_preprocess_func(function): Function that defines how to preprocess weight before quantization. Using this
                can quickly test if user's preprocess method works or not. The function's input
                is non-quantized weight and function returns processed weight to be quantized. If None, the weight will
                be quantized directly.
                Default is None.
        act_preprocess_func(function): Function that defines how to preprocess activation before quantization. Using this
                can quickly test if user's preprocess method works or not. The function's input
                is non-quantized activation and function returns processed activation to be quantized. If None, the activation will
                be quantized directly.
                Default is None.
        optimizer_func(function): Fuction return a optimizer. When 'is_test' is False and user want to use self-defined 
            quantization function and preprocess function, this function must be set. Default is None.
        exe(paddle.static.Executor): If user want to use self-defined quantization function and preprocess function, exe must be set for
                initialization. Default is None.
        return_program(bool): If user want return value is a Program rather than Compiled Program, This argument should be set True.
                Default is False.
    Returns:
        paddle.static.CompiledProgram | paddle.static.Program: Program with quantization and dequantization ``operators``
    """

    scope = paddle.static.global_scope() if not scope else scope
    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("quant_aware config {}".format(config))

    main_graph = IrGraph(core.Graph(program.desc), for_test=for_test)

    transform_pass_ops = []
    quant_dequant_ops = []
    for op_type in config['quantize_op_types']:
        if op_type in TRANSFORM_PASS_OP_TYPES:
            transform_pass_ops.append(op_type)
        elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
            quant_dequant_ops.append(op_type)
    if len(transform_pass_ops) > 0:
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            weight_bits=config['weight_bits'],
            activation_bits=config['activation_bits'],
            activation_quantize_type=config['activation_quantize_type'],
            weight_quantize_type=config['weight_quantize_type'],
            window_size=config['window_size'],
            moving_rate=config['moving_rate'],
            quantizable_op_type=transform_pass_ops,
            skip_pattern=config['not_quant_pattern'],
            weight_quantize_func=weight_quantize_func,
            act_quantize_func=act_quantize_func,
            weight_preprocess_func=weight_preprocess_func,
            act_preprocess_func=act_preprocess_func,
            optimizer_func=optimizer_func,
            executor=executor)

        transform_pass.apply(main_graph)

    if len(quant_dequant_ops) > 0:
        quant_dequant_pass = AddQuantDequantPass(
            scope=scope,
            place=place,
            moving_rate=config['moving_rate'],
            quant_bits=config['activation_bits'],
            skip_pattern=config['not_quant_pattern'],
            quantizable_op_type=quant_dequant_ops)
        quant_dequant_pass.apply(main_graph)

    out_scale_training_pass = OutScaleForTrainingPass(
        scope=scope, place=place, moving_rate=config['moving_rate'])
    out_scale_training_pass.apply(main_graph)

    if (weight_preprocess_func is not None or
            act_preprocess_func is not None) and not for_test:
        _logger.info(
            "When a preprocess_func is used in quant_aware, Need to save a mapping table to match variable names in the convert phase."
        )
        _logger.info("The mapping table is saved as '{}'.".format(
            VARS_MAPPING_TABLE))
        save_dict(main_graph.out_node_mapping_table)

    if for_test or return_program:
        quant_program = main_graph.to_program()
    else:
        quant_program = paddle.static.CompiledProgram(main_graph.graph)
    return quant_program


def quant_post_static(
        executor,
        model_dir,
        quantize_model_path,
        batch_generator=None,
        sample_generator=None,
        model_filename=None,
        params_filename=None,
        save_model_filename='__model__',
        save_params_filename='__params__',
        batch_size=16,
        batch_nums=None,
        scope=None,
        algo='hist',
        hist_percent=0.9999,
        bias_correction=False,
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        activation_quantize_type='range_abs_max',
        weight_quantize_type='channel_wise_abs_max',
        optimize_model=False,
        is_use_cache_file=False,
        cache_dir="./temp_post_training"):
    """
    The function utilizes static post training quantization method to
    quantize the fp32 model. It uses calibrate data to calculate the
    scale factor of quantized variables, and inserts fake quantization
    and dequantization operators to obtain the quantized model.

    Args:
        executor(paddle.static.Executor): The executor to load, run and save the 
            quantized model.
        model_dir(str): The path of fp32 model that will be quantized, and 
            the model and params that saved by ``paddle.static.io.save_inference_model`` 
            are under the path.
        quantize_model_path(str): The path to save quantized model using api
            ``paddle.static.io.save_inference_model``.
        batch_generator(Python Generator): The batch generator provides 
                calibrate data for DataLoader, and it returns a batch every
                time. For sample_generator and batch_generator, only one
                can be set. Beisdes, batch_generator supports lod tensor.
        sample_generator(Python Generator): The sample generator provides 
            calibrate data for DataLoader, and it only returns a sample every time.
        model_filename(str, optional): The name of model file. If parameters 
            are saved in separate files, set it as 'None'. Default: 'None'.
        params_filename(str, optional): The name of params file.
                When all parameters are saved in a single file, set it 
                as filename. If parameters are saved in separate files, 
                set it as 'None'. Default : 'None'.
        save_model_filename(str): The name of model file to save the quantized inference program.  Default: '__model__'.
        save_params_filename(str): The name of file to save all related parameters. 
                If it is set None, parameters will be saved in separate files. Default: '__params__'.
        batch_size(int, optional): The batch size of DataLoader, default is 16.
        batch_nums(int, optional): If batch_nums is not None, the number of calibrate 
                        data is 'batch_size*batch_nums'. If batch_nums is None, use all data
                        generated by sample_generator  as calibrate data.
        scope(paddle.static.Scope, optional): The scope to run program, use it to load 
                        and save variables. If scope is None, will use paddle.static.global_scope().
        algo(str, optional): If algo='KL', use KL-divergenc method to 
                        get the scale factor. If algo='hist', use the hist_percent of histogram 
                        to get the scale factor. If algo='mse', search for the best scale factor which
                        makes the mse loss minimal. Use one batch of data for mse is enough. If 
                        algo='avg', use the average of abs_max values  to get the scale factor. If 
                        algo='abs_max', use abs_max method to get the scale factor. Default: 'hist'.
        hist_percent(float, optional): The percentile of histogram for algo hist.Default:0.9999.
        bias_correction(bool, optional): Bias correction method of https://arxiv.org/abs/1810.05723.
                        Default: False.
        quantizable_op_type(list[str], optional): The list of op types
                        that will be quantized. Default: ["conv2d", "depthwise_conv2d", 
                        "mul"].
        weight_bits(int, optional): quantization bit number for weights.
        activation_bits(int): quantization bit number for activation.
	activation_quantize_type(str): quantization type for activation,
                now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
                This parameter only specifies the fake ops in quantized model.
                If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
                obtained by post training quantization in fake ops. If it
                is 'abs_max', the scale will not be saved in fake ops.
        weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. Compared to 'abs_max',
                the model accuracy is usually higher when using 'channel_wise_abs_max'.
        is_full_quantize(bool): if True, apply quantization to all supported quantizable op type.
                        If False, only apply quantization to the input quantizable_op_type. Default is False.
        optimize_model(bool, optional): If set optimize_model as True, it applies some 
                passes to optimize the model before quantization. So far, the place of
                executor must be cpu it supports fusing batch_norm into convs.
        is_use_cache_file(bool): This param is deprecated.
        cache_dir(str): This param is deprecated.
    
    Returns:
        None
    """
    post_training_quantization = PostTrainingQuantization(
        executor=executor,
        sample_generator=sample_generator,
        batch_generator=batch_generator,
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        batch_size=batch_size,
        batch_nums=batch_nums,
        scope=scope,
        algo=algo,
        hist_percent=hist_percent,
        bias_correction=bias_correction,
        quantizable_op_type=quantizable_op_type,
        is_full_quantize=is_full_quantize,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        activation_quantize_type=activation_quantize_type,
        weight_quantize_type=weight_quantize_type,
        optimize_model=optimize_model)
    post_training_quantization.quantize()
    post_training_quantization.save_quantized_model(
        quantize_model_path,
        model_filename=save_model_filename,
        params_filename=save_params_filename)


# We have changed the quant_post to quant_post_static.
# For compatibility, we keep quant_post api for now, and it will be
# deprecated in the future.
quant_post = quant_post_static


def convert(program, place, config=None, scope=None, save_int8=False):
    """
    convert quantized and well-trained ``program`` to final  quantized
    ``program``that can be used to  save ``inference model``.
    
    Args:
        program(paddle.static.Program): quantized and well-trained ``test program``.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents
                the executor run on which device.
        config(dict, optional): configs for convert. if set None, will use
                default config. It must be same with config that used in
                'quant_aware'. Default is None.
        scope(paddle.static.Scope, optional):  Scope records the mapping between
                variable names and variables, similar to brackets in
                programming languages. Usually users can use
                `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.
                When ``None`` will use 
                `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_
                . Default: ``None``.
        save_int8: Whether to return ``program`` which model parameters'
                dtype is ``int8``. This parameter can only be used to
                get model size. Default: ``False``.

    Returns:
        Tuple : freezed program which can be used for inference.
                when ``save_int8`` is False, return ``freezed_program(paddle.static.Program)``.
                when ``save_int8`` is True, return ``freezed_program(paddle.static.Program)``
                and ``freezed_program_int8(paddle.static.Program)``
    """
    scope = paddle.static.global_scope() if not scope else scope

    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("convert config {}".format(config))
    test_graph = IrGraph(core.Graph(program.desc), for_test=True)

    out_scale_infer_pass = OutScaleForInferencePass(scope=scope)
    out_scale_infer_pass.apply(test_graph)

    # Freeze the graph after training by adjusting the quantize
    # operators' order for the inference.
    freeze_pass = QuantizationFreezePass(
        scope=scope,
        place=place,
        weight_bits=config['weight_bits'],
        activation_bits=config['activation_bits'],
        weight_quantize_type=config['weight_quantize_type'])

    if os.path.exists(VARS_MAPPING_TABLE):
        test_graph.out_node_mapping_table = load_dict()

    freeze_pass.apply(test_graph)
    freezed_program = test_graph.to_program()

    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(scope=scope, place=place)
        convert_int8_pass.apply(test_graph)
        freezed_program_int8 = test_graph.to_program()
        return freezed_program, freezed_program_int8
    else:
        return freezed_program


def quant_post_dynamic(model_dir,
                       save_model_dir,
                       model_filename=None,
                       params_filename=None,
                       save_model_filename=None,
                       save_params_filename=None,
                       quantizable_op_type=["conv2d", "mul"],
                       weight_bits=8,
                       generate_test_model=False):
    '''
    The function utilizes static post training quantization method to
    quantize the fp32 model. In details, it quantizes the weight of some
    ops from float32 to int8/16. For the quantized model, there are two
    kinds of calculation method in the reference stage. Firstly, the
    quantized weight will be dequantized to float32, and then apply the
    float32 calculation. Secondly, collect the quantized scales of the
    inputs, and then apply the int8 calculation.
        
    Args:
        model_dir(str): The path of the fp32 model that will be quantized,
                and the model and params files are under the path.
        save_model_dir(str): The path to save the quantized model.
        model_filename(str, optional): The name of file used to load the
                inference program. If it is None, the default filename
                '__model__' will be used. Default is 'None'.
        params_filename(str, optional): The name of file used to load all
                parameters. When all parameters were saved in a single
                binary file, set it as the real filename. If parameters
                were saved in separate files, set it as 'None'. Default is
                'None'.
        save_model_dir(str): The path used to save the quantized model.
        save_model_filename(str, optional): The name of file to 
                save the inference program. If it is None, the default 
                filename '__model__' will be used. Default is 'None'.
        save_params_filename(str, optional): The name of file to 
                save all parameters. If it is None, parameters were 
                saved in separate files. If it is not None, all 
                parameters were saved in a single binary file.
        quantizable_op_type(list[str], optional): The list of ops 
                that will be quantized, and the quantized ops should be
                contained in ["conv2d", "depthwise_conv2d", "mul"]. 
                Default is ["conv2d", "depthwise_conv2d", "mul"].
        weight_bits(int, optional): The bits for the quantized weight, 
                and it should be 8 or 16. Default is 8.
        generate_test_model(bool, optional): If set generate_test_model 
                as True, it saves a fake quantized model, in which the weights 
                are quantized and dequantized. We can use PaddlePaddle to load 
                the fake quantized model and test the accuracy on GPU or CPU.
    '''

    weight_quant = WeightQuantization(
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename)

    weight_quant.quantize_weight_to_int(
        save_model_dir=save_model_dir,
        save_model_filename=save_model_filename,
        save_params_filename=save_params_filename,
        quantizable_op_type=quantizable_op_type,
        weight_bits=weight_bits,
        generate_test_model=generate_test_model)


# We have changed the quant_post_only_weight to quant_post_dynamic.
# For compatibility, we keep quant_post_only_weight api for now,
# and it will be deprecated in the future.
quant_post_only_weight = quant_post_dynamic
