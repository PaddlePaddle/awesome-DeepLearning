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

import argparse
import os
import os.path as osp
import sys

import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import get_config


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo export model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument("-p",
                        "--pretrained_params",
                        default='./best.pdparams',
                        type=str,
                        help='params path')
    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        default="./inference",
                        help='output path')

    return parser.parse_args()


def trim_config(cfg):
    """
    Reuse the trainging config will bring useless attributes, such as: backbone.pretrained model.
    and some build phase attributes should be overrided, such as: backbone.num_seg.
    Trim it here.
    """
    model_name = cfg.model_name
    if cfg.MODEL.get('backbone') and cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ""  # not ued when inference

    return cfg, model_name


def get_input_spec(cfg, model_name):
    if model_name in ['ppTSM', 'TSM']:
        input_spec = [[
            InputSpec(
                shape=[None, cfg.num_seg, 3, cfg.target_size, cfg.target_size],
                dtype='float32'),
        ]]
    elif model_name in ['TSN', 'ppTSN']:
        input_spec = [[
            InputSpec(shape=[
                None, cfg.num_seg * 10, 3, cfg.target_size, cfg.target_size
            ],
                      dtype='float32'),
        ]]
    elif model_name in ['BMN']:
        input_spec = [[
            InputSpec(shape=[None, cfg.feat_dim, cfg.tscale],
                      dtype='float32',
                      name='feat_input'),
        ]]
    elif model_name in ['TimeSformer', 'ppTimeSformer']:
        input_spec = [[
            InputSpec(shape=[
                None, 3, cfg.num_seg * 3, cfg.target_size, cfg.target_size
            ],
                      dtype='float32'),
        ]]
    elif model_name in ['VideoSwin']:
        input_spec = [[
            InputSpec(shape=[
                None, 3, cfg.num_seg * cfg.seg_len * 1, cfg.target_size,
                cfg.target_size
            ],
                      dtype='float32'),
        ]]
    elif model_name in ['VideoSwin_TableTennis']:
        input_spec = [[
            InputSpec(shape=[
                None, 3, cfg.num_seg * cfg.seg_len * 3, cfg.target_size,
                cfg.target_size
            ],
                      dtype='float32'),
        ]]
    elif model_name in ['AttentionLSTM']:
        input_spec = [[
            InputSpec(shape=[None, cfg.embedding_size, cfg.feature_dims[0]],
                      dtype='float32'),  # for rgb_data
            InputSpec(shape=[
                None,
            ], dtype='int64'),  # for rgb_len
            InputSpec(shape=[None, cfg.embedding_size, cfg.feature_dims[0]],
                      dtype='float32'),  # for rgb_mask
            InputSpec(shape=[None, cfg.embedding_size, cfg.feature_dims[1]],
                      dtype='float32'),  # for audio_data
            InputSpec(shape=[
                None,
            ], dtype='int64'),  # for audio_len
            InputSpec(shape=[None, cfg.embedding_size, cfg.feature_dims[1]],
                      dtype='float32'),  # for audio_mask
        ]]
    elif model_name in ['SlowFast']:
        input_spec = [[
            InputSpec(shape=[
                None, 3, cfg.num_frames // cfg.alpha, cfg.target_size,
                cfg.target_size
            ],
                      dtype='float32',
                      name='slow_input'),
            InputSpec(shape=[
                None, 3, cfg.num_frames, cfg.target_size, cfg.target_size
            ],
                      dtype='float32',
                      name='fast_input'),
        ]]
    elif model_name in ['STGCN', 'AGCN']:
        input_spec = [[
            InputSpec(shape=[
                None, cfg.num_channels, cfg.window_size, cfg.vertex_nums,
                cfg.person_nums
            ],
                      dtype='float32'),
        ]]
    elif model_name in ['TransNetV2']:
        input_spec = [[
            InputSpec(shape=[
                None,
                cfg.num_frames,
                cfg.height,
                cfg.width,
                cfg.num_channels,
            ],
                      dtype='float32'),
        ]]
    elif model_name in ['ADDS']:
        input_spec = [[
            InputSpec(shape=[None, cfg.num_channels, cfg.height, cfg.width],
                      dtype='float32'),
        ]]
    elif model_name in ['AVA_SlowFast_FastRcnn']:
        input_spec = [[
            InputSpec(shape=[
                None, 3, cfg.num_frames // cfg.alpha, cfg.target_size,
                cfg.target_size
            ],
                      dtype='float32',
                      name='slow_input'),
            InputSpec(shape=[
                None, 3, cfg.num_frames, cfg.target_size, cfg.target_size
            ],
                      dtype='float32',
                      name='fast_input'),
            InputSpec(shape=[None, None, 4], dtype='float32', name='proposals'),
            InputSpec(shape=[None, 2], dtype='float32', name='img_shape')
        ]]
    return input_spec


def main():
    args = parse_args()
    cfg, model_name = trim_config(get_config(args.config, show=False))
    print(f"Building model({model_name})...")
    model = build_model(cfg.MODEL)
    assert osp.isfile(
        args.pretrained_params
    ), f"pretrained params ({args.pretrained_params} is not a file path.)"

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    print(f"Loading params from ({args.pretrained_params})...")
    params = paddle.load(args.pretrained_params)
    model.set_dict(params)

    model.eval()

    input_spec = get_input_spec(cfg.INFERENCE, model_name)
    model = to_static(model, input_spec=input_spec)
    paddle.jit.save(model, osp.join(args.output_path, model_name))
    print(
        f"model ({model_name}) has been already saved in ({args.output_path}).")


if __name__ == "__main__":
    main()
