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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    #   jsonfile: 预测输出的结果文件，例如: bbox.json.
    #   style: COCO 数据集的, can be `bbox` and `proposal`.
    #   coco_gt: 使用 COCO API 对标注文件进行解析后返回的对象，计算方式: coco_gt = COCO(anno_file)
    #   anno_file: COCO 格式的标注文件.
    #   max_dets: 预定义的检测模型最大的输出个数.

    assert coco_gt != None or anno_file != None

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    # 计算mAP
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # 返回mAP结果
    return coco_eval.stats
