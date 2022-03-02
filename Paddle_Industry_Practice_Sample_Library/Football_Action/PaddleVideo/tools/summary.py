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
import sys
import os.path as osp

import paddle
import paddle.nn.functional as F
from paddle.jit import to_static
import paddleslim

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import get_config


def parse_args():

    parser = argparse.ArgumentParser("PaddleVideo Summary")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_seg", type=int, default=8)
    parser.add_argument("--FLOPs",
                        action="store_true",
                        help="whether to print FLOPs")

    return parser.parse_args()


def _trim(cfg, args):
    """
    Reuse the trainging config will bring useless attribute, such as: backbone.pretrained model. Trim it here.
    """
    model_name = cfg.model_name
    cfg = cfg.MODEL
    cfg.backbone.pretrained = ""

    if 'num_seg' in cfg.backbone:
        cfg.backbone.num_seg = args.num_seg
    return cfg, model_name


def main():
    args = parse_args()
    cfg, model_name = _trim(get_config(args.config, show=False), args)
    print(f"Building model({model_name})...")
    model = build_model(cfg)

    img_size = args.img_size
    num_seg = args.num_seg
    #NOTE: only support tsm now, will refine soon
    params_info = paddle.summary(model, (1, 1, num_seg, 3, img_size, img_size))
    print(params_info)

    if args.FLOPs:
        flops_info = paddleslim.analysis.flops(model, [1, 1, num_seg, 3, img_size, img_size])
        print(flops_info)


if __name__ == "__main__":
    main()
