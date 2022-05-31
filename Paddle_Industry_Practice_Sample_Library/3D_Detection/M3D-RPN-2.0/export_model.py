import os
import sys
import argparse
import ast
import logging
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from models import *
from easydict import EasyDict as edict
from lib.rpn_util import *

sys.path.append(os.getcwd())
import lib.core as core
from lib.util import *
import pdb

import paddle
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)



def parse_args():
    """parse"""
    parser = argparse.ArgumentParser("M3D-RPN train script")
    parser.add_argument("--conf_path", type=str, default='', help="config.pkl")
    parser.add_argument(
        '--weights_path', type=str, default='', help='weights save path')

    parser.add_argument(
        '--backbone',
        type=str,
        default='DenseNet121',
        help='backbone model to train, default DenseNet121')

    parser.add_argument(
        '--data_dir', type=str, default='dataset', help='dataset directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    conf = edict(pickle_read(args.conf_path))
    conf.pretrained = None
    src_path = os.path.join('.', 'models', conf.model + '_infer.py')
    print(src_path)
    train_model = absolute_import(src_path)
    train_model = train_model.build(conf, args.backbone, 'eval')
    train_model.eval()
    train_model.phase = "eval"
    paddle.disable_static()
    params = paddle.load(args.weights_path)
    train_model.set_dict(params, use_structured_name=True)
    train_model.eval()

    input=[
            paddle.static.InputSpec(
                shape=[1, 3, 512, 1760], name="input", dtype='float32')
        ]

 
    new_net = paddle.jit.to_static(train_model, input_spec=input)
    paddle.jit.save(new_net, './inference/model')
