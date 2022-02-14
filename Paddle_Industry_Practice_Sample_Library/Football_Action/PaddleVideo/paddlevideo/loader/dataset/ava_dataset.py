# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import copy
import random
import numpy as np
import sys
import os
import pickle
from datetime import datetime
from ...metrics.ava_utils import ava_evaluate_results
from ..registry import DATASETS
from .base import BaseDataset
from collections import defaultdict


@DATASETS.register()
class AVADataset(BaseDataset):
    """AVA dataset for spatial temporal detection.
    the dataset loads raw frames, bounding boxes, proposals and applies
    transformations to return the frame tensors and other information.
    """

    _FPS = 30

    def __init__(self,
                 pipeline,
                 file_path=None,
                 exclude_file=None,
                 label_file=None,
                 suffix='{:05}.jpg',
                 proposal_file=None,
                 person_det_score_thr=0.9,
                 num_classes=81,
                 data_prefix=None,
                 test_mode=False,
                 num_max_proposals=1000,
                 timestamp_start=900,
                 timestamp_end=1800):
        self.custom_classes = None
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.num_classes = num_classes
        self.suffix = suffix
        self.num_max_proposals = num_max_proposals
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        super().__init__(
            file_path,
            pipeline,
            data_prefix,
            test_mode,
        )
        if self.proposal_file is not None:
            self.proposals = self._load(self.proposal_file)
        else:
            self.proposals = None
        if not test_mode:
            valid_indexes = self.filter_exclude_file()
            self.info = self.info = [self.info[i] for i in valid_indexes]

    def _load(self, path):
        f = open(path, 'rb')
        res = pickle.load(f)
        f.close()
        return res

    def parse_img_record(self, img_records):
        bboxes, labels, entity_ids = [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)
            selected_records = list(
                filter(
                    lambda x: np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            num_selected_records = len(selected_records)
            img_records = list(
                filter(
                    lambda x: not np.array_equal(x['entity_box'], img_record[
                        'entity_box']), img_records))
            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label'] for selected_record in selected_records
            ])

            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def filter_exclude_file(self):
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.info)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, video_info in enumerate(self.info):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (video_info['video_id'] == video_id
                            and video_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break
        return valid_indexes

    def load_file(self):
        """Load index file to get video information."""
        info = []
        records_dict_by_img = defaultdict(list)
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(',')

                video_id = line_split[0]
                timestamp = int(line_split[1])
                img_key = f'{video_id},{timestamp:04d}'

                entity_box = np.array(list(map(float, line_split[2:6])))
                label = int(line_split[6])
                entity_id = int(line_split[7])
                shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                             self._FPS)

                video_info = dict(video_id=video_id,
                                  timestamp=timestamp,
                                  entity_box=entity_box,
                                  label=label,
                                  entity_id=entity_id,
                                  shot_info=shot_info)
                records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            bboxes, labels, entity_ids = self.parse_img_record(
                records_dict_by_img[img_key])
            ann = dict(gt_bboxes=bboxes,
                       gt_labels=labels,
                       entity_ids=entity_ids)
            frame_dir = video_id
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(frame_dir=frame_dir,
                              video_id=video_id,
                              timestamp=int(timestamp),
                              img_key=img_key,
                              shot_info=shot_info,
                              fps=self._FPS,
                              ann=ann)
            info.append(video_info)

        return info

    def prepare_train(self, idx):
        results = copy.deepcopy(self.info[idx])
        img_key = results['img_key']

        results['suffix'] = self.suffix
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                results['proposals'] = np.array([[0, 0, 1, 1]])
                results['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals[:, :4]
                    results['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    results['proposals'] = proposals

        ann = results.pop('ann')
        results['gt_bboxes'] = ann['gt_bboxes']
        results['gt_labels'] = ann['gt_labels']
        results['entity_ids'] = ann['entity_ids']

        #ret = self.pipeline(results, "")
        ret = self.pipeline(results)
        #padding for dataloader
        len_proposals = ret['proposals'].shape[0]
        len_gt_bboxes = ret['gt_bboxes'].shape[0]
        len_gt_labels = ret['gt_labels'].shape[0]
        len_scores = ret['scores'].shape[0]
        len_entity_ids = ret['entity_ids'].shape[0]
        padding_len = 128
        ret['proposals'] = self.my_padding_2d(ret['proposals'], padding_len)
        ret['gt_bboxes'] = self.my_padding_2d(ret['gt_bboxes'], padding_len)
        ret['gt_labels'] = self.my_padding_2d(ret['gt_labels'], padding_len)
        ret['scores'] = self.my_padding_1d(ret['scores'], padding_len)
        ret['entity_ids'] = self.my_padding_1d(ret['entity_ids'], padding_len)
        return ret['imgs'][0], ret['imgs'][1], ret['proposals'], ret[
            'gt_bboxes'], ret['gt_labels'], ret['scores'], ret[
                'entity_ids'], np.array(
                    ret['img_shape'], dtype=int
                ), idx, len_proposals, len_gt_bboxes, len_gt_labels, len_scores, len_entity_ids

    def my_padding_2d(self, feat, max_len):
        feat_add = np.zeros((max_len - feat.shape[0], feat.shape[1]),
                            dtype=np.float32)
        feat_pad = np.concatenate((feat, feat_add), axis=0)
        return feat_pad

    def my_padding_1d(self, feat, max_len):
        feat_add = np.zeros((max_len - feat.shape[0]), dtype=np.float32)
        feat_pad = np.concatenate((feat, feat_add), axis=0)
        return feat_pad

    def prepare_test(self, idx):
        return self.prepare_train(idx)

    def evaluate(self, results):
        return ava_evaluate_results(self.info, len(self), results,
                                    self.custom_classes, self.label_file,
                                    self.file_path, self.exclude_file)
