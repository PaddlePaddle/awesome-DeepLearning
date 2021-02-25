# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
from .box_iou_xyxy import box_iou_xyxy


# 非极大值抑制
def nms(bboxes, scores, score_thresh, nms_thresh):
    """
    nms
    """
    inds = np.argsort(scores)
    inds = inds[::-1]
    keep_inds = []
    while (len(inds) > 0):
        cur_ind = inds[0]
        cur_score = scores[cur_ind]
        # if score of the box is less than score_thresh, just drop it
        if cur_score < score_thresh:
            break

        keep = True
        for ind in keep_inds:
            current_box = bboxes[cur_ind]
            remain_box = bboxes[ind]
            iou = box_iou_xyxy(current_box, remain_box)
            if iou > nms_thresh:
                keep = False
                break
        if keep:
            keep_inds.append(cur_ind)
        inds = inds[1:]

    return np.array(keep_inds)


# 画图展示目标物体边界框
# 定义画矩形框的程序    
def draw_rectangle(currentAxis,
                   bbox,
                   edgecolor='k',
                   facecolor='y',
                   fill=False,
                   linestyle='-'):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0] + 1,
        bbox[3] - bbox[1] + 1,
        linewidth=1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        fill=fill,
        linestyle=linestyle)
    currentAxis.add_patch(rect)


if __name__ == '__main__':
    plt.figure(figsize=(10, 10))

    filename = '/home/aistudio/work/images/section3/000000086956.jpg'
    im = imread(filename)
    plt.imshow(im)

    currentAxis = plt.gca()

    boxes = np.array(
        [[4.21716537e+01, 1.28230896e+02, 2.26547668e+02, 6.00434631e+02],
         [3.18562988e+02, 1.23168472e+02, 4.79000000e+02, 6.05688416e+02],
         [2.62704697e+01, 1.39430557e+02, 2.20587097e+02, 6.38959656e+02],
         [4.24965363e+01, 1.42706665e+02, 2.25955185e+02, 6.35671204e+02],
         [2.37462646e+02, 1.35731537e+02, 4.79000000e+02, 6.31451294e+02],
         [3.19390472e+02, 1.29295090e+02, 4.79000000e+02, 6.33003845e+02],
         [3.28933838e+02, 1.22736115e+02, 4.79000000e+02, 6.39000000e+02],
         [4.44292603e+01, 1.70438187e+02, 2.26841858e+02, 6.39000000e+02],
         [2.17988785e+02, 3.02472412e+02, 4.06062927e+02, 6.29106628e+02],
         [2.00241089e+02, 3.23755096e+02, 3.96929321e+02, 6.36386108e+02],
         [2.14310303e+02, 3.23443665e+02, 4.06732849e+02, 6.35775269e+02]])

    scores = np.array([
        0.5247661, 0.51759845, 0.86075854, 0.9910175, 0.39170712, 0.9297706,
        0.5115228, 0.270992, 0.19087596, 0.64201415, 0.879036
    ])

    left_ind = np.where((boxes[:, 0] < 60) * (boxes[:, 0] > 20))
    left_boxes = boxes[left_ind]
    left_scores = scores[left_ind]

    colors = ['r', 'g', 'b', 'k']

    # 画出最终保留的预测框
    inds = nms(boxes, scores, score_thresh=0.01, nms_thresh=0.5)
    # 打印最终保留的预测框是哪几个
    print(inds)
    for i in range(len(inds)):
        box = boxes[inds[i]]
        draw_rectangle(currentAxis, box, edgecolor=colors[i])
