#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import numpy as np
from ..registry import PIPELINES
"""pipeline ops for Activity Net.
"""


@PIPELINES.register()
class LoadFeat(object):
    def __init__(self, feat_path):
        self.feat_path = feat_path

    def __call__(self, results):
        video_name = results['video_name']
        file_name = video_name + ".npy"
        file_path = os.path.join(self.feat_path, file_name)
        #TODO: check path
        video_feat = np.load(file_path)
        video_feat = video_feat.T
        video_feat = video_feat.astype("float32")
        results['video_feat'] = video_feat
        return results


@PIPELINES.register()
class GetMatchMap(object):
    def __init__(self, tscale):
        self.tscale = tscale
        self.tgap = 1. / self.tscale

    def __call__(self, results):
        match_map = []
        for idx in range(self.tscale):
            tmp_match_window = []
            xmin = self.tgap * idx
            for jdx in range(1, self.tscale + 1):
                xmax = xmin + self.tgap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)
        match_map = np.transpose(match_map, [1, 0, 2])
        match_map = np.reshape(match_map, [-1, 2])

        anchor_xmin = [self.tgap * i for i in range(self.tscale)]
        anchor_xmax = [self.tgap * i for i in range(1, self.tscale + 1)]

        results['match_map'] = match_map
        results['anchor_xmin'] = anchor_xmin
        results['anchor_xmax'] = anchor_xmax
        return results


@PIPELINES.register()
class GetVideoLabel(object):
    def __init__(self, tscale, dscale, datatype="float32"):
        self.tscale = tscale
        self.dscale = dscale
        self.tgap = 1. / self.tscale
        self.datatype = datatype

    def iou_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        union_len = len_anchors - inter_len + box_max - box_min
        jaccard = np.divide(inter_len, union_len)
        return jaccard

    def ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        """Compute intersection between score a box and the anchors.
        """
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def __call__(self, results):
        video_info = results['video_info']
        match_map = results['match_map']
        anchor_xmin = results['anchor_xmin']
        anchor_xmax = results['anchor_xmax']

        video_second = video_info['duration_second']
        video_labels = video_info['annotations']

        gt_bbox = []
        gt_iou_map = []
        for gt in video_labels:
            tmp_start = max(min(1, gt["segment"][0] / video_second), 0)
            tmp_end = max(min(1, gt["segment"][1] / video_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = self.iou_with_anchors(match_map[:, 0],
                                                   match_map[:, 1], tmp_start,
                                                   tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.dscale, self.tscale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_len_small = 3 * self.tgap
        gt_start_bboxs = np.stack(
            (gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack(
            (gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(
                np.max(
                    self.ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                          gt_start_bboxs[:, 0],
                                          gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(
                np.max(
                    self.ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                          gt_end_bboxs[:, 0], gt_end_bboxs[:,
                                                                           1])))

        gt_start = np.array(match_score_start)
        gt_end = np.array(match_score_end)

        results['gt_iou_map'] = gt_iou_map.astype(self.datatype)
        results['gt_start'] = gt_start.astype(self.datatype)
        results['gt_end'] = gt_end.astype(self.datatype)
        return results
