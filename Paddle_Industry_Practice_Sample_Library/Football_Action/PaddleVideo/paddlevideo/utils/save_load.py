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
import os
import os.path as osp
import time

import paddle
import paddle.nn.functional as F
from paddlevideo.utils import get_logger, main_only
from tqdm import tqdm


def pretrain_swin_param_trans(model, state_dicts):
    # delete classifier's params
    if 'head.fc' + '.weight' in state_dicts:
        del state_dicts['head.fc' + '.weight']
    if 'head.fc' + '.bias' in state_dicts:
        del state_dicts['head.fc' + '.bias']

    state_dicts = {
        k.replace('backbone.', ''): v
        for k, v in state_dicts.items()
    }

    if len(state_dicts) == len(model.state_dict()):
        print("Load 3D weights")
        return state_dicts

    print("Load 2D weights")
    relative_position_index_keys = [
        k for k in state_dicts.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del state_dicts[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dicts.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dicts[k]

    state_dicts['patch_embed.proj.weight'] = state_dicts[
        'patch_embed.proj.weight'].unsqueeze(2).tile(
            [1, 1, model.patch_size[0], 1, 1]) / model.patch_size[0]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dicts.keys() if "relative_position_bias_table" in k
    ]
    total_len = len(relative_position_bias_table_keys)
    with tqdm(total=total_len,
              position=1,
              bar_format='{desc}',
              desc="Loading weights") as desc:
        for key in tqdm(relative_position_bias_table_keys,
                        total=total_len,
                        position=0):
            relative_position_bias_table_pretrained = state_dicts[key]
            relative_position_bias_table_current = model.state_dict()[key]
            L1, nH1 = relative_position_bias_table_pretrained.shape
            L2, nH2 = relative_position_bias_table_current.shape
            L2 = (2 * model.window_size[1] - 1) * (2 * model.window_size[2] - 1)
            wd = model.window_size[0]
            if nH1 != nH2:
                desc.set_description(f"Error in loading {key}, skip")
            else:
                if L1 != L2:
                    S1 = int(L1**0.5)
                    relative_position_bias_table_pretrained_resized = paddle.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.transpose(
                            [1, 0]).reshape([1, nH1, S1, S1]),
                        size=(2 * model.window_size[1] - 1,
                              2 * model.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.reshape(
                        [nH2, L2]).transpose([1, 0])
                desc.set_description(f"Loading {key}")
            state_dicts[key] = relative_position_bias_table_pretrained.tile(
                [2 * wd - 1, 1])
            time.sleep(0.01)
    ret_str = "loading {:<20d} weights completed.".format(
        len(model.state_dict()))
    desc.set_description(ret_str)
    return state_dicts


def pretrain_vit_param_trans(model, state_dicts, num_patches, num_seg,
                             attention_type):
    """
    Convert ViT's pre-trained model parameters to a parameter dictionary that matches the existing model
    """
    if 'head' + '.weight' in state_dicts:
        del state_dicts['head' + '.weight']
    if 'head' + '.bias' in state_dicts:
        del state_dicts['head' + '.bias']

    total_len = len(model.state_dict())
    if num_patches + 1 != state_dicts['pos_embed'].shape[1]:
        pos_embed = state_dicts['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0,
                                    1:, :].unsqueeze(0).unsqueeze(1).transpose(
                                        (0, 1, 3, 2))
        new_pos_embed = F.interpolate(other_pos_embed,
                                      size=(other_pos_embed.shape[-2],
                                            num_patches),
                                      mode='nearest')
        new_pos_embed = new_pos_embed.squeeze(0).transpose((0, 2, 1))
        new_pos_embed = paddle.concat((cls_pos_embed, new_pos_embed), axis=1)
        state_dicts['pos_embed'] = new_pos_embed
        time.sleep(0.01)

    if 'time_embed' in state_dicts and num_seg != state_dicts[
            'time_embed'].shape[1]:
        time_embed = state_dicts['time_embed'].transpose((0, 2, 1)).unsqueeze(0)
        new_time_embed = F.interpolate(time_embed,
                                       size=(time_embed.shape[-2], num_seg),
                                       mode='nearest')
        state_dicts['time_embed'] = new_time_embed.squeeze(0).transpose(
            (0, 2, 1))
        time.sleep(0.01)
    with tqdm(total=total_len,
              position=1,
              bar_format='{desc}',
              desc="Loading weights") as desc:
        if attention_type == 'divided_space_time':
            new_state_dicts = state_dicts.copy()
            for key in tqdm(state_dicts):
                if 'blocks' in key and 'attn' in key:
                    desc.set_description("Loading %s" % key)
                    new_key = key.replace('attn', 'temporal_attn')
                    if not new_key in state_dicts:
                        new_state_dicts[new_key] = state_dicts[key]
                    else:
                        new_state_dicts[new_key] = state_dicts[new_key]
                if 'blocks' in key and 'norm1' in key:
                    desc.set_description("Loading %s" % key)
                    new_key = key.replace('norm1', 'temporal_norm1')
                    if not new_key in state_dicts:
                        new_state_dicts[new_key] = state_dicts[key]
                    else:
                        new_state_dicts[new_key] = state_dicts[new_key]
                time.sleep(0.01)
    ret_str = "loading {:<20d} weights completed.".format(
        len(model.state_dict()))
    desc.set_description(ret_str)
    return new_state_dicts


def pretrain_resnet18_param_trans(model, loaded_dict):
    encoder_dict = model.encoder.state_dict()
    pose_encoder_dict = model.pose_encoder.state_dict()

    names = ['encoder.', 'encoder_day.', 'encoder_night.']
    for name in names:
        total_len = len(loaded_dict.items())
        with tqdm(total=total_len,
                  position=1,
                  bar_format='{desc}',
                  desc="Loading weights") as desc:
            for key, value in tqdm(loaded_dict.items(),
                                   total=total_len,
                                   position=0):
                key = str(name + key)
                if key in encoder_dict:
                    encoder_dict[key] = value
                    desc.set_description('Loading %s' % key)
                time.sleep(0.01)

    num_input_images = 2
    loaded_dict['conv1.weight'] = paddle.concat(
        [loaded_dict['conv1.weight']] * num_input_images, 1) / num_input_images
    total_len = len(loaded_dict.items())
    with tqdm(total=total_len,
              position=1,
              bar_format='{desc}',
              desc="Loading weights") as desc:
        for name, value in tqdm(loaded_dict.items(),
                                total=total_len,
                                position=0):
            name = str('encoder.' + name)
            if name in pose_encoder_dict:
                pose_encoder_dict[name] = value
                desc.set_description('Loading %s' % key)
            time.sleep(0.01)
        ret_str = "loading {:<20d} weights completed.".format(
            len(model.state_dict()))
        desc.set_description(ret_str)
    return encoder_dict, pose_encoder_dict


#XXX(shipping): maybe need load N times because of different cards have different params.
@main_only
def load_ckpt(model, weight_path, **kargs):
    """
    1. Load pre-trained model parameters
    2. Extract and convert from the pre-trained model to the parameters
    required by the existing model
    3. Load the converted parameters of the existing model
    """
    #model.set_state_dict(state_dict)

    if not osp.isfile(weight_path):
        raise IOError(f'{weight_path} is not a checkpoint file')
    #state_dicts = load(weight_path)

    logger = get_logger("paddlevideo")
    state_dicts = paddle.load(weight_path)
    if 'ResnetEncoder' in str(model):
        encoder_dict, pose_encoder_dict = pretrain_resnet18_param_trans(
            model, state_dicts)
        model.encoder.load_dict(encoder_dict)
        model.pose_encoder.load_dict(pose_encoder_dict)
        tmp = model.state_dict()
    elif "VisionTransformer" in str(model):  # For TimeSformer case
        tmp = pretrain_vit_param_trans(model, state_dicts, kargs['num_patches'],
                                       kargs['num_seg'],
                                       kargs['attention_type'])
    elif 'SwinTransformer3D' in str(model):
        tmp = pretrain_swin_param_trans(model, state_dicts)
    else:
        tmp = {}
        total_len = len(model.state_dict())
        with tqdm(total=total_len,
                  position=1,
                  bar_format='{desc}',
                  desc="Loading weights") as desc:
            for item in tqdm(model.state_dict(), total=total_len, position=0):
                name = item
                desc.set_description('Loading %s' % name)
                if name not in state_dicts:  # Convert from non-parallel model
                    if str('backbone.' + name) in state_dicts:
                        tmp[name] = state_dicts['backbone.' + name]
                else:  # Convert from parallel model
                    tmp[name] = state_dicts[name]
                time.sleep(0.01)
        ret_str = "loading {:<20d} weights completed.".format(
            len(model.state_dict()))
        desc.set_description(ret_str)
    model.set_state_dict(tmp)


def mkdir(dir):
    if not os.path.exists(dir):
        # avoid error when train with multiple gpus
        try:
            os.makedirs(dir)
        except:
            pass


"""
def save(state_dicts, file_name):
    def convert(state_dict):
        model_dict = {}

        for k, v in state_dict.items():
            if isinstance(
                    v,
                (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(
                v,
            (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v

    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f, protocol=2)
"""


@main_only
def save(obj, path):
    paddle.save(obj, path)


def load(file_name):
    if not osp.isfile(file_name):
        raise IOError(f'{file_name} not exist')
    return paddle.load(file_name)
