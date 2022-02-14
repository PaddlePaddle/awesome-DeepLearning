"""
# @File  : process_result.py  
# @Author: macaihong
# @Date  : 2019/12/15
# @Desc  :
"""

import sys
import os
import re
import numpy as np
import pickle
import json
import logger

logger = logger.Logger()


def get_data_res(label_map, data, topk):
    """get_data_res"""
    sum_vid = len(data)
    video_result = []
    for i in range(sum_vid):
        vid_name = data[i][0][0]
        # true_label predict_start predict_end predict_score predict_len gt_iou gt_start gt_ioa
        feature_start_id = float(data[i][0][1]['start'])
        feature_end_id = float(data[i][0][1]['end'])
        feature_stage1_score = data[i][0][1]['score']
        predict_res = []
        for k in range(topk):
            score_top = data[i][1][k]
            labelid_top = data[i][2][k]
            label_iou = data[i][3]
            labelname_top = label_map[str(labelid_top)]
            video_result.append([feature_start_id, feature_end_id, labelid_top, labelname_top, score_top, label_iou])
    return video_result


def base_nms(bboxes, thresh, delta=0, nms_id=2):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[vid, label, st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    """
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, nms_id]
    """

    t1 = np.array([max(0, x[0] - delta) for x in bboxes])
    t2 = np.array([x[1] + delta for x in bboxes])
    scores = np.array([x[nms_id] for x in bboxes])

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]
    return [bboxes[i] for i in keep]


def process_proposal(source_prop_box, min_frame_thread=5, nms_thresh=0.7, score_thresh=0.01):
    """process_video_prop"""
    prop_box = []
    for items in source_prop_box:
        start_frame = float(items[0])
        end_frame = float(items[1])
        score = float(items[2])
        if end_frame - start_frame < min_frame_thread or score < score_thresh:
            continue
        prop_box.append([start_frame, end_frame, score])

    prop_box_keep = base_nms(prop_box, nms_thresh)

    prop_res = []
    for res in prop_box_keep:
        prop_res.append({'start': res[0], 'end': res[1], 'score': res[2]})

    return prop_res


def process_video_classify(video_prop, fps, score_thread, iou_thread, \
                           nms_id=5, nms_thread=0.01, nms_delta=10, backgroundid=0):
    """process_video_classify"""
    prop_filter = []
    for item in video_prop:
        if item[2] == backgroundid:
            continue
        prop_filter.append(item)

    # prop_filter = sorted(prop_filter, key=lambda x: x[nms_id], reverse=True)
    prop_filter = base_nms(prop_filter, nms_thread, nms_delta, nms_id)
    prop_filter = sorted(prop_filter, key=lambda x: x[0])

    video_results = []
    for item in prop_filter:
        start_sec = item[0] / fps
        end_sec = item[1] / fps

        start_id_frame = item[0]
        end_id_frame = item[1]
        # start_time = "%02d:%02d:%02d" % ((start_id_frame / fps) / 3600, \
        #     ((start_id_frame / fps) % 3600) / 60, (start_id_frame / fps) % 60)
        # end_time = "%02d:%02d:%02d" % ((end_id_frame / fps) / 3600, \
        #     ((end_id_frame / fps) % 3600) / 60, (end_id_frame / fps) % 60)
        start_time = int(start_id_frame / fps)
        end_time = int(end_id_frame / fps)

        label_id = item[2]
        label_name = item[3]
        label_classify_score = item[4]
        label_iou_score = item[5]
        if label_classify_score > score_thread and label_iou_score > iou_thread:
            video_results.append({"start_time": start_time,
                                  "end_time": end_time,
                                  "label_id": label_id,
                                  "label_name": label_name,
                                  "classify_score": label_classify_score,
                                  "iou_score": label_iou_score})

    return video_results


def get_action_result(result_info, label_map_file, fps, score_thread=0, \
                      iou_thread=0, nms_id=5, nms_thread=0.01, frame_offset=10, topk=1):
    """get_action_result"""

    label_map = json.load(open(label_map_file, 'r', encoding='utf-8'))

    org_result = get_data_res(label_map, result_info, topk)
    nms_result = process_video_classify(org_result, fps, score_thread, iou_thread, nms_id, nms_thread, frame_offset)

    return nms_result
