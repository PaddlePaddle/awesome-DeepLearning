# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from .utils import foreground2background, global_matching_for_eval, local_matching, calculate_attention_head_for_eval
from ...registry import SEGMENT
from .base import BaseSegment
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@SEGMENT.register()
class CFBI(BaseSegment):
    """CFBI model framework."""
    def __init__(self, backbone=None, head=None, loss=None):
        super().__init__(backbone, head, loss)
        x1 = paddle.zeros([3, 1, 1, 1])
        self.bg_bias = paddle.create_parameter(
            shape=x1.shape,
            dtype=x1.dtype,
            default_initializer=nn.initializer.Assign(x1))
        self.fg_bias = paddle.create_parameter(
            shape=x1.shape,
            dtype=x1.dtype,
            default_initializer=nn.initializer.Assign(x1))
        self.epsilon = 1e-05

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output.
        """
        self.test_mode = True
        ref_embeddings, ref_masks, prev_embedding, prev_mask, current_frame, pred_size, gt_ids = data_batch
        current_frame_embedding_4x, current_frame_embedding_8x, current_frame_embedding_16x, \
        current_low_level = self.backbone(current_frame)

        current_frame_embedding = [
            current_frame_embedding_4x, current_frame_embedding_8x,
            current_frame_embedding_16x
        ]

        if prev_embedding is None:
            return None, current_frame_embedding
        else:
            bs, c, h, w = current_frame_embedding_4x.shape

            tmp_dic, _ = self.before_seghead_process(
                ref_embeddings,
                prev_embedding,
                current_frame_embedding,
                ref_masks,
                prev_mask,
                gt_ids,
                current_low_level=current_low_level,
            )
            all_pred = []
            for i in range(bs):
                pred = tmp_dic[i]

                pred = F.interpolate(pred,
                                     size=[pred_size[0], pred_size[1]],
                                     mode='bilinear',
                                     align_corners=True)
                all_pred.append(pred)
            all_pred = paddle.concat(all_pred, axis=0)
            all_pred = F.softmax(all_pred, axis=1)
            return all_pred, current_frame_embedding

    def before_seghead_process(self,
                               ref_frame_embeddings=None,
                               previous_frame_embeddings=None,
                               current_frame_embeddings=None,
                               ref_frame_labels=None,
                               previous_frame_mask=None,
                               gt_ids=None,
                               current_low_level=None):
        """ process befor segmentation head"""
        TEST_GLOBAL_MATCHING_CHUNK = [4, 1, 1]
        TEST_GLOBAL_ATROUS_RATE = [2, 1, 1]
        TRAIN_LOCAL_ATROUS_RATE = [2, 1, 1]
        TEST_LOCAL_ATROUS_RATE = [2, 1, 1]
        MODEL_FLOAT16_MATCHING = False
        TEST_GLOBAL_MATCHING_MIN_PIXEL = 100
        MODEL_MULTI_LOCAL_DISTANCE = [[4, 8, 12, 16, 20, 24],
                                      [2, 4, 6, 8, 10, 12], [2, 4, 6, 8, 10]]
        TRAIN_LOCAL_PARALLEL = True
        TEST_LOCAL_PARALLEL = True
        MODEL_MATCHING_BACKGROUND = True
        MODEL_SEMANTIC_MATCHING_DIM = [32, 64, 128]

        dic_tmp = []
        boards = {}
        scale_ref_frame_labels = []
        scale_previous_frame_labels = []
        for current_frame_embedding in current_frame_embeddings:
            bs, c, h, w = current_frame_embedding.shape
            if not self.test_mode:
                raise NotImplementedError
            else:
                ref_frame_embeddings = list(zip(*ref_frame_embeddings))
                all_scale_ref_frame_label = []
                for ref_frame_label in ref_frame_labels:
                    scale_ref_frame_label = paddle.cast(F.interpolate(
                        paddle.cast(ref_frame_label, dtype="float32"),
                        size=(h, w),
                        mode='nearest'),
                                                        dtype="int32")
                    all_scale_ref_frame_label.append(scale_ref_frame_label)
                scale_ref_frame_labels.append(all_scale_ref_frame_label)
            scale_previous_frame_label = paddle.cast(F.interpolate(
                paddle.cast(previous_frame_mask, dtype="float32"),
                size=(h, w),
                mode='nearest'),
                                                     dtype="int32")
            scale_previous_frame_labels.append(scale_previous_frame_label)
        for n in range(bs):
            ref_obj_ids = paddle.reshape(
                paddle.cast(paddle.arange(0,
                                          np.array(gt_ids)[n] + 1),
                            dtype="int32"), [-1, 1, 1, 1])
            obj_num = ref_obj_ids.shape[0]
            low_level_feat = paddle.unsqueeze(current_low_level[n], axis=0)
            all_CE_input = []
            all_attention_head = []
            for scale_idx, current_frame_embedding, ref_frame_embedding, previous_frame_embedding, \
                scale_ref_frame_label, scale_previous_frame_label in zip(range(3), \
                    current_frame_embeddings, ref_frame_embeddings, previous_frame_embeddings, \
                    scale_ref_frame_labels, scale_previous_frame_labels):
                #Prepare
                seq_current_frame_embedding = current_frame_embedding[n]
                seq_prev_frame_embedding = previous_frame_embedding[n]
                seq_previous_frame_label = paddle.cast(
                    (paddle.cast(scale_previous_frame_label[n], dtype="int32")
                     == ref_obj_ids),
                    dtype="float32")
                if np.array(gt_ids)[n] > 0:
                    dis_bias = paddle.concat([
                        paddle.unsqueeze(self.bg_bias[scale_idx], axis=0),
                        paddle.expand(
                            paddle.unsqueeze(self.fg_bias[scale_idx], axis=0),
                            [np.array(gt_ids)[n], -1, -1, -1])
                    ],
                                             axis=0)
                else:
                    dis_bias = paddle.unsqueeze(self.bg_bias[scale_idx], axis=0)
                #Global FG map
                matching_dim = MODEL_SEMANTIC_MATCHING_DIM[scale_idx]
                seq_current_frame_embedding_for_matching = paddle.transpose(
                    seq_current_frame_embedding[:matching_dim], [1, 2, 0])

                if not self.test_mode:
                    raise NotImplementedError
                else:
                    all_scale_ref_frame_label = scale_ref_frame_label
                    all_ref_frame_embedding = ref_frame_embedding
                    all_reference_embeddings = []
                    all_reference_labels = []
                    seq_ref_frame_labels = []
                    count = 0
                    for idx in range(len(all_scale_ref_frame_label)):

                        ref_frame_embedding = all_ref_frame_embedding[idx]
                        scale_ref_frame_label = all_scale_ref_frame_label[idx]

                        seq_ref_frame_embedding = ref_frame_embedding[n]
                        seq_ref_frame_embedding = paddle.transpose(
                            seq_ref_frame_embedding, [1, 2, 0])
                        seq_ref_frame_label = paddle.cast(
                            (paddle.cast(scale_ref_frame_label[n],
                                         dtype="int32") == ref_obj_ids),
                            dtype="float32")
                        seq_ref_frame_labels.append(seq_ref_frame_label)
                        seq_ref_frame_label = paddle.transpose(
                            paddle.squeeze(seq_ref_frame_label, axis=1),
                            [1, 2, 0])
                        all_reference_embeddings.append(
                            seq_ref_frame_embedding[:, :, :matching_dim])
                        all_reference_labels.append(seq_ref_frame_label)
                    global_matching_fg = global_matching_for_eval(
                        all_reference_embeddings=all_reference_embeddings,
                        query_embeddings=
                        seq_current_frame_embedding_for_matching,
                        all_reference_labels=all_reference_labels,
                        n_chunks=TEST_GLOBAL_MATCHING_CHUNK[scale_idx],
                        dis_bias=dis_bias,
                        atrous_rate=TEST_GLOBAL_ATROUS_RATE[scale_idx],
                        use_float16=MODEL_FLOAT16_MATCHING,
                        atrous_obj_pixel_num=TEST_GLOBAL_MATCHING_MIN_PIXEL)

                # Local FG map
                seq_prev_frame_embedding_for_matching = paddle.transpose(
                    seq_prev_frame_embedding[:matching_dim], [1, 2, 0])
                seq_previous_frame_label_for_matching = paddle.transpose(
                    paddle.squeeze(seq_previous_frame_label, axis=1), [1, 2, 0])
                local_matching_fg = local_matching(
                    prev_frame_embedding=seq_prev_frame_embedding_for_matching,
                    query_embedding=seq_current_frame_embedding_for_matching,
                    prev_frame_labels=seq_previous_frame_label_for_matching,
                    multi_local_distance=MODEL_MULTI_LOCAL_DISTANCE[scale_idx],
                    dis_bias=dis_bias,
                    atrous_rate=TRAIN_LOCAL_ATROUS_RATE[scale_idx] if
                    not self.test_mode else TEST_LOCAL_ATROUS_RATE[scale_idx],
                    use_float16=MODEL_FLOAT16_MATCHING,
                    allow_downsample=False,
                    allow_parallel=TRAIN_LOCAL_PARALLEL
                    if not self.test_mode else TEST_LOCAL_PARALLEL)

                #Aggregate Pixel-level Matching
                to_cat_global_matching_fg = paddle.transpose(
                    paddle.squeeze(global_matching_fg, axis=0), [2, 3, 0, 1])
                to_cat_local_matching_fg = paddle.transpose(
                    paddle.squeeze(local_matching_fg, axis=0), [2, 3, 0, 1])
                all_to_cat = [
                    to_cat_global_matching_fg, to_cat_local_matching_fg,
                    seq_previous_frame_label
                ]

                #Global and Local BG map
                if MODEL_MATCHING_BACKGROUND:
                    to_cat_global_matching_bg = foreground2background(
                        to_cat_global_matching_fg,
                        np.array(gt_ids)[n] + 1)
                    reshaped_prev_nn_feature_n = paddle.unsqueeze(
                        paddle.transpose(to_cat_local_matching_fg,
                                         [0, 2, 3, 1]),
                        axis=1)
                    to_cat_local_matching_bg = foreground2background(
                        reshaped_prev_nn_feature_n,
                        np.array(gt_ids)[n] + 1)
                    to_cat_local_matching_bg = paddle.squeeze(paddle.transpose(
                        to_cat_local_matching_bg, [0, 4, 2, 3, 1]),
                                                              axis=-1)
                    all_to_cat += [
                        to_cat_local_matching_bg, to_cat_global_matching_bg
                    ]

                to_cat_current_frame_embedding = paddle.expand(
                    paddle.unsqueeze(current_frame_embedding[n], axis=0),
                    [obj_num, -1, -1, -1])
                to_cat_prev_frame_embedding = paddle.expand(
                    paddle.unsqueeze(previous_frame_embedding[n], axis=0),
                    [obj_num, -1, -1, -1])
                to_cat_prev_frame_embedding_fg = to_cat_prev_frame_embedding * seq_previous_frame_label
                to_cat_prev_frame_embedding_bg = to_cat_prev_frame_embedding * (
                    1 - seq_previous_frame_label)
                all_to_cat += [
                    to_cat_current_frame_embedding,
                    to_cat_prev_frame_embedding_fg,
                    to_cat_prev_frame_embedding_bg
                ]

                CE_input = paddle.concat(all_to_cat, axis=1)
                #Instance-level Attention
                if not self.test_mode:
                    raise NotImplementedError
                else:
                    attention_head = calculate_attention_head_for_eval(
                        all_ref_frame_embedding,
                        seq_ref_frame_labels,
                        paddle.expand(
                            paddle.unsqueeze(previous_frame_embedding[n],
                                             axis=0), [obj_num, -1, -1, -1]),
                        seq_previous_frame_label,
                        epsilon=self.epsilon)

                all_CE_input.append(CE_input)
                all_attention_head.append(attention_head)

            #Collaborative Ensembler
            pred = self.head(all_CE_input, all_attention_head, low_level_feat)
            dic_tmp.append(pred)

        return dic_tmp, boards
