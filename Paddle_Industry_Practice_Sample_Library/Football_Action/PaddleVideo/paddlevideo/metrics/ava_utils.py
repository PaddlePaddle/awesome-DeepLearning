# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import csv
import heapq
import logging
import time
from collections import defaultdict
from .ava_evaluation import object_detection_evaluation as det_eval
from .ava_evaluation import standard_fields
from .recall import eval_recalls
import shutil
import pickle
import time
import os
import os.path as osp
from paddlevideo.utils import get_logger, get_dist_info
import paddle.distributed as dist
import sys
import numpy as np
from pathlib import Path
from datetime import datetime


def det2csv(info, dataset_len, results, custom_classes):
    csv_results = []
    for idx in range(dataset_len):
        video_id = info[idx]['video_id']
        timestamp = info[idx]['timestamp']

        result = results[idx]
        for label, _ in enumerate(result):
            for bbox in result[label]:
                if type(bbox) == paddle.Tensor:
                    bbox = bbox.numpy()
                
                bbox_ = tuple(bbox.tolist())
                if custom_classes is not None:
                    actual_label = custom_classes[label + 1]
                else:
                    actual_label = label + 1
                csv_results.append((
                    video_id,
                    timestamp,
                ) + bbox_[:4] + (actual_label, ) + bbox_[4:])
    return csv_results


# results is organized by class
def results2csv(info, dataset_len, results, out_file, custom_classes=None):
    if isinstance(results[0], list):
        csv_results = det2csv(info, dataset_len, results, custom_classes)

    # save space for float
    def tostr(item):
        if isinstance(item, float):
            return f'{item:.3f}'
        return str(item)

    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(lambda x: tostr(x), csv_result)))
            f.write('\n')


def print_time(message, start):
    print('==> %g seconds to %s' % (time.time() - start, message))


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{int(timestamp):04d}'


def read_csv(csv_file, class_whitelist=None, capacity=0):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
        csv_file: A file object.
        class_whitelist: If provided, boxes corresponding to (integer) class
        labels not in this set are skipped.
        capacity: Maximum number of labeled boxes allowed for each example.
        Default is 0 where there is no limit.

    Returns:
        boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
        labels: A dictionary mapping each unique image key (string) to a list
        of integer class lables, matching the corresponding box in `boxes`.
        scores: A dictionary mapping each unique image key (string) to a list
        of score values lables, matching the corresponding label in `labels`.
        If scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], 'Wrong number of columns: ' + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue

        score = 1.0
        if len(row) == 8:
            score = float(row[7])
        if capacity < 1 or len(entries[image_key]) < capacity:
            heapq.heappush(entries[image_key],
                           (score, action_id, y1, x1, y2, x2))
        elif score > entries[image_key][0][0]:
            heapq.heapreplace(entries[image_key],
                              (score, action_id, y1, x1, y2, x2))
    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        for item in entry:
            score, action_id, y1, x1, y2, x2 = item
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    print_time('read file ' + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g.
        "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, 'Expected only 2 columns, got: ' + row
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the
        object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


# Seems there is at most 100 detections for each image
def ava_eval(result_file,
             result_type,
             label_file,
             ann_file,
             exclude_file,
             max_dets=(100, ),
             verbose=True,
             custom_classes=None):

    assert result_type in ['mAP']
    start = time.time()
    categories, class_whitelist = read_labelmap(open(label_file))

    if custom_classes is not None:
        custom_classes = custom_classes[1:]
        assert set(custom_classes).issubset(set(class_whitelist))
        class_whitelist = custom_classes
        categories = [cat for cat in categories if cat['id'] in custom_classes]

    # loading gt, do not need gt score
    gt_boxes, gt_labels, _ = read_csv(open(ann_file), class_whitelist, 0)
    if verbose:
        print_time('Reading detection results', start)

    if exclude_file is not None:
        excluded_keys = read_exclusions(open(exclude_file))
    else:
        excluded_keys = list()

    start = time.time()
    boxes, labels, scores = read_csv(open(result_file), class_whitelist, 0)
    if verbose:
        print_time('Reading detection results', start)

    if result_type == 'proposal':
        gts = [
            np.array(gt_boxes[image_key], dtype=float) for image_key in gt_boxes
        ]
        proposals = []
        for image_key in gt_boxes:
            if image_key in boxes:
                proposals.append(
                    np.concatenate(
                        (np.array(boxes[image_key], dtype=float),
                         np.array(scores[image_key], dtype=float)[:, None]),
                        axis=1))
            else:
                # if no corresponding proposal, add a fake one
                proposals.append(np.array([0, 0, 1, 1, 1]))

        # Proposals used here are with scores
        recalls = eval_recalls(gts, proposals, np.array(max_dets),
                               np.arange(0.5, 0.96, 0.05))
        ar = recalls.mean(axis=1)
        ret = {}
        for i, num in enumerate(max_dets):
            print(f'Recall@0.5@{num}\t={recalls[i, 0]:.4f}')
            print(f'AR@{num}\t={ar[i]:.4f}')
            ret[f'Recall@0.5@{num}'] = recalls[i, 0]
            ret[f'AR@{num}'] = ar[i]
        return ret

    if result_type == 'mAP':
        pascal_evaluator = det_eval.PascalDetectionEvaluator(categories)

        start = time.time()
        for image_key in gt_boxes:
            if verbose and image_key in excluded_keys:
                logging.info(
                    'Found excluded timestamp in detections: %s.'
                    'It will be ignored.', image_key)
                continue
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(gt_boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                    np.array(gt_labels[image_key], dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(gt_boxes[image_key]), dtype=bool)
                })
        if verbose:
            print_time('Convert groundtruth', start)

        start = time.time()
        for image_key in boxes:
            if verbose and image_key in excluded_keys:
                logging.info(
                    'Found excluded timestamp in detections: %s.'
                    'It will be ignored.', image_key)
                continue
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float)
                })
        if verbose:
            print_time('convert detections', start)

        start = time.time()
        metrics = pascal_evaluator.evaluate()
        if verbose:
            print_time('run_evaluator', start)
        for display_name in metrics:
            print(f'{display_name}=\t{metrics[display_name]}')
        ret = {
            display_name: metrics[display_name]
            for display_name in metrics if 'ByCategory' not in display_name
        }
        return ret


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def dump_to_fileobj(obj, file, **kwargs):
    kwargs.setdefault('protocol', 2)
    pickle.dump(obj, file, **kwargs)


def dump_to_path(obj, filepath, mode='wb'):
    with open(filepath, mode) as f:
        dump_to_fileobj(obj, f)


def load_from_fileobj(file, **kwargs):
    return pickle.load(file, **kwargs)


def load_from_path(filepath, mode='rb'):
    with open(filepath, mode) as f:
        return load_from_fileobj(f)


def collect_results_cpu(result_part, size):
    """Collect results in cpu mode.
    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.
    """
    tmpdir = osp.join('./', 'collect_results_cpu')
    #1. load results of all parts from tmp dir
    mkdir_or_exist(tmpdir)
    rank, world_size = get_dist_info()
    dump_to_path(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    if rank != 0:
        return None
    #2. collect all parts
    while 1:
        all_exist = True
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            if not Path(part_file).exists():
                all_exist = False
        if all_exist:
            break
        else:
            time.sleep(60)
    time.sleep(120)
    #3. load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(load_from_path(part_file))
    #4. sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:
                                      size]  #the dataloader may pad some samples
    #5. remove results of all parts from tmp dir, avoid dump_file fail to tmp dir when dir not exists.
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        os.remove(part_file)

    return ordered_results


def ava_evaluate_results(info, dataset_len, results, custom_classes, label_file,
                         file_path, exclude_file):
    # need to create a temp result file
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_file = f'AVA_{time_now}_result.csv'
    results2csv(info, dataset_len, results, temp_file)
    ret = {}
    eval_result = ava_eval(
        temp_file,
        'mAP',
        label_file,
        file_path,  #ann_file,
        exclude_file,
        custom_classes=custom_classes)
    ret.update(eval_result)

    os.remove(temp_file)

    return ret
