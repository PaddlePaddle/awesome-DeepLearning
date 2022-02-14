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

import os
import paddle
import zipfile
import time
from PIL import Image

from paddle.io import DataLoader

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@METRIC.register
class VOSMetric(BaseMetric):
    def __init__(self,
                 data_size,
                 batch_size,
                 result_root,
                 zip_dir,
                 log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.video_num = 0
        self.total_time = 0
        self.total_frame = 0
        self.total_sfps = 0
        self.total_video_num = data_size
        self.count = 0
        self.result_root = result_root
        self.zip_dir = zip_dir

    def update(self, batch_id, data, model):
        """update metrics during each iter
        """
        self.video_num += 1
        seq_dataset = data
        seq_name = seq_dataset.seq_name

        logger.info('Prcessing Seq {} [{}/{}]:'.format(seq_name, self.video_num,
                                                       self.total_video_num))
        seq_dataloader = DataLoader(seq_dataset,
                                    return_list=True,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0)
        seq_total_time = 0
        seq_total_frame = 0
        ref_embeddings = []
        ref_masks = []
        prev_embedding = []
        prev_mask = []
        with paddle.no_grad():
            for frame_idx, samples in enumerate(seq_dataloader):
                time_start = time.time()
                all_preds = []
                join_label = None
                for aug_idx in range(len(samples)):
                    if len(ref_embeddings) <= aug_idx:
                        ref_embeddings.append([])
                        ref_masks.append([])
                        prev_embedding.append(None)
                        prev_mask.append(None)

                    sample = samples[aug_idx]
                    ref_emb = ref_embeddings[aug_idx]
                    ref_m = ref_masks[aug_idx]
                    prev_emb = prev_embedding[aug_idx]
                    prev_m = prev_mask[aug_idx]

                    current_img = sample['current_img']
                    if 'current_label' in sample.keys():
                        current_label = sample['current_label']
                        current_label = paddle.to_tensor(current_label)
                    else:
                        current_label = None

                    obj_num = sample['meta']['obj_num']
                    imgname = sample['meta']['current_name']
                    ori_height = sample['meta']['height']
                    ori_width = sample['meta']['width']
                    current_img = current_img
                    obj_num = obj_num
                    bs, _, h, w = current_img.shape
                    data_batch = [
                        ref_emb, ref_m, prev_emb, prev_m, current_img,
                        [ori_height, ori_width], obj_num
                    ]

                    all_pred, current_embedding = model(data_batch, mode='test')

                    if frame_idx == 0:
                        if current_label is None:
                            logger.info(
                                "No first frame label in Seq {}.".format(
                                    seq_name))
                        ref_embeddings[aug_idx].append(current_embedding)
                        ref_masks[aug_idx].append(current_label)

                        prev_embedding[aug_idx] = current_embedding
                        prev_mask[aug_idx] = current_label
                    else:
                        if sample['meta']['flip']:  #False
                            all_pred = self.flip_tensor(all_pred, 3)
                        #  In YouTube-VOS, not all the objects appear in the first frame for the first time. Thus, we
                        #  have to introduce new labels for new objects, if necessary.
                        if not sample['meta']['flip'] and not (
                                current_label is None) and join_label is None:
                            join_label = paddle.cast(current_label,
                                                     dtype='int64')
                        all_preds.append(all_pred)
                        if current_label is not None:
                            ref_embeddings[aug_idx].append(current_embedding)
                        prev_embedding[aug_idx] = current_embedding

                if frame_idx > 0:
                    all_preds = paddle.concat(all_preds, axis=0)
                    all_preds = paddle.mean(
                        all_preds, axis=0)  #average results if augmentation
                    pred_label = paddle.argmax(all_preds, axis=0)
                    if join_label is not None:
                        join_label = paddle.squeeze(paddle.squeeze(join_label,
                                                                   axis=0),
                                                    axis=0)
                        keep = paddle.cast((join_label == 0), dtype="int64")
                        pred_label = pred_label * keep + join_label * (1 - keep)
                        pred_label = pred_label
                    current_label = paddle.reshape(
                        pred_label, shape=[1, 1, ori_height, ori_width])
                    flip_pred_label = self.flip_tensor(pred_label, 1)
                    flip_current_label = paddle.reshape(
                        flip_pred_label, shape=[1, 1, ori_height, ori_width])

                    for aug_idx in range(len(samples)):
                        if join_label is not None:
                            if samples[aug_idx]['meta']['flip']:
                                ref_masks[aug_idx].append(flip_current_label)
                            else:
                                ref_masks[aug_idx].append(current_label)
                        if samples[aug_idx]['meta']['flip']:
                            prev_mask[aug_idx] = flip_current_label
                        else:
                            prev_mask[
                                aug_idx] = current_label  #update prev_mask

                    one_frametime = time.time() - time_start
                    seq_total_time += one_frametime
                    seq_total_frame += 1
                    obj_num = obj_num.numpy()[0].item()
                    logger.info('Frame: {}, Obj Num: {}, Time: {}'.format(
                        imgname[0], obj_num, one_frametime))
                    self.save_mask(
                        pred_label,
                        os.path.join(self.result_root, seq_name,
                                     imgname[0].split('.')[0] + '.png'))
                else:
                    one_frametime = time.time() - time_start
                    seq_total_time += one_frametime
                    logger.info('Ref Frame: {}, Time: {}'.format(
                        imgname[0], one_frametime))

            del (ref_embeddings)
            del (ref_masks)
            del (prev_embedding)
            del (prev_mask)
            del (seq_dataset)
            del (seq_dataloader)

        seq_avg_time_per_frame = seq_total_time / seq_total_frame
        self.total_time += seq_total_time
        self.total_frame += seq_total_frame
        total_avg_time_per_frame = self.total_time / self.total_frame
        self.total_sfps += seq_avg_time_per_frame
        avg_sfps = self.total_sfps / (batch_id + 1)
        logger.info("Seq {} FPS: {}, Total FPS: {}, FPS per Seq: {}".format(
            seq_name, 1. / seq_avg_time_per_frame,
            1. / total_avg_time_per_frame, 1. / avg_sfps))

    def flip_tensor(self, tensor, dim=0):
        inv_idx = paddle.cast(paddle.arange(tensor.shape[dim] - 1, -1, -1),
                              dtype="int64")
        tensor = paddle.index_select(x=tensor, index=inv_idx, axis=dim)
        return tensor

    def save_mask(self, mask_tensor, path):
        _palette = [
            0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128,
            0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191,
            128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64,
            0, 128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22,
            22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27,
            28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33,
            33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39,
            39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44,
            45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50,
            50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56,
            56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61,
            62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67,
            67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73,
            73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78,
            79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84,
            84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90,
            90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95,
            96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101,
            101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105,
            105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109,
            110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114,
            114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118,
            118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122,
            123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127,
            127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131,
            131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135,
            136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140,
            140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144,
            144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148,
            149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153,
            153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157,
            157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161,
            162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166,
            166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170,
            170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174,
            175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179,
            179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183,
            183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187,
            188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192,
            192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196,
            196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200,
            201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205,
            205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209,
            209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213,
            214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218,
            218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222,
            222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226,
            227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231,
            231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235,
            235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239,
            240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244,
            244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248,
            248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252,
            253, 253, 253, 254, 254, 254, 255, 255, 255
        ]
        mask = mask_tensor.cpu().numpy().astype('uint8')
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(_palette)
        mask.save(path)

    def zip_folder(self, source_folder, zip_dir):
        f = zipfile.ZipFile(zip_dir, 'w', zipfile.ZIP_DEFLATED)
        pre_len = len(os.path.dirname(source_folder))
        for dirpath, dirnames, filenames in os.walk(source_folder):
            for filename in filenames:
                pathfile = os.path.join(dirpath, filename)
                arcname = pathfile[pre_len:].strip(os.path.sep)
                f.write(pathfile, arcname)
        f.close()

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        self.zip_folder(self.result_root, self.zip_dir)
        logger.info('Save result to {}.'.format(self.zip_dir))
