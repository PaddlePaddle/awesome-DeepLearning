# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import paddle
import os, sys
import copy as cp
import cv2
import math

import ppdet

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.modeling.builder import build_model
from paddlevideo.utils import get_config
from paddlevideo.loader.builder import build_dataloader, build_dataset, build_pipeline
from paddlevideo.metrics.ava_utils import read_labelmap

import time
from os import path as osp
import numpy as np
from paddlevideo.utils import get_config
import pickle

from paddlevideo.utils import (get_logger, load, mkdir, save)
import shutil

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


# annotations is pred results
def visualize(frames, annotations, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.
    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5，目前不能大于5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    # proposals被归一化需要还原真实坐标值
    scale_ratio = np.array([w, h, w, h])

    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def frame_extraction(video_path, target_dir):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, '{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)

    FPS = int(vid.get(5))

    frames = []
    frame_paths = []

    flag, frame = vid.read()
    index = 1
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(index)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        index += 1
        flag, frame = vid.read()
    return frame_paths, frames, FPS


def parse_args():

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')

    parser.add_argument('--video_path', help='video file/url')

    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')

    #detection_model_name
    parser.add_argument('--detection_model_name',
                        help='the name of detection model ')
    # detection_model_weights
    parser.add_argument('--detection_model_weights',
                        help='the weights path of detection model ')

    # params for predict
    parser.add_argument('--out-filename',
                        default='ava_det_demo.mp4',
                        help='output filename')
    parser.add_argument('--predict-stepsize',
                        default=8,
                        type=int,
                        help='give out a prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=4,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument('--output-fps',
                        default=6,
                        type=int,
                        help='the fps of demo video output')

    return parser.parse_args()


# 一帧的结果。根据概率大小进行排序
def pack_result(human_detection, result):
    """Short summary.
    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    results = []
    if result is None:
        return None

    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])

        results.append((prop, [x[0] for x in res], [x[1] for x in res]))

    return results


# 构造数据处理需要的results
def get_timestep_result(frame_dir, timestamp, clip_len, frame_interval, FPS):
    result = {}

    result["frame_dir"] = frame_dir

    frame_num = len(os.listdir(frame_dir))

    dir_name = frame_dir.split("/")[-1]
    result["video_id"] = dir_name

    result['timestamp'] = timestamp

    timestamp_str = '{:04d}'.format(timestamp)
    img_key = dir_name + "," + timestamp_str
    result['img_key'] = img_key

    result['shot_info'] = (1, frame_num)
    result['fps'] = FPS

    result['suffix'] = '{:05}.jpg'

    result['timestamp_start'] = 1
    result['timestamp_end'] = int(frame_num / result['fps'])

    return result


def detection_inference(frame_paths, output_dir, model_name, weights_path):
    """Detect human boxes given frame paths.
    Args:
        frame_paths (list[str]): The paths of frames to do detection inference.
    Returns:
        list[np.ndarray]: The human detection results.
    """

    detection_cfg = ppdet.model_zoo.get_config_file(model_name)
    detection_cfg = ppdet.core.workspace.load_config(detection_cfg)
    detection_trainer = ppdet.engine.Trainer(detection_cfg, mode='test')
    detection_trainer.load_weights(weights_path)

    print('Performing Human Detection for each frame')

    detection_trainer.predict(frame_paths, output_dir=output_dir, save_txt=True)

    print("finish object detection")

    results = []

    for frame_path in frame_paths:
        (file_dir, file_name) = os.path.split(frame_path)
        (file_path, ext) = os.path.splitext(frame_path)

        txt_file_name = file_name.replace(ext, ".txt")
        txt_path = os.path.join(output_dir, txt_file_name)
        results.append(txt_path)

    return results


def get_detection_result(txt_file_path, img_h, img_w, person_det_score_thr):
    """
    根据检测结果文件得到图像中人的检测框(proposals)和置信度（scores）
    txt_file_path:检测结果存放路径
    img_h:图像高度
    img_w:图像宽度
    """

    proposals = []
    scores = []

    with open(txt_file_path, 'r') as detection_file:
        lines = detection_file.readlines()
        for line in lines:  # person 0.9842637181282043 0.0 469.1407470703125 944.7770385742188 831.806396484375
            items = line.split(" ")
            if items[0] != 'person':  #只要人
                continue

            score = items[1]

            if (float)(score) < person_det_score_thr:
                continue

            x1 = (float(items[2])) / img_w
            y1 = ((float)(items[3])) / img_h
            box_w = ((float)(items[4]))
            box_h = ((float)(items[5]))

            x2 = (float(items[2]) + box_w) / img_w
            y2 = (float(items[3]) + box_h) / img_h

            scores.append(score)

            proposals.append([x1, y1, x2, y2])

    return np.array(proposals), np.array(scores)


@paddle.no_grad()
def main(args):
    config = get_config(args.config, show=False)  #parse config file

    # extract frames from video
    video_path = args.video_path
    frame_dir = 'tmp_frames'
    frame_paths, frames, FPS = frame_extraction(video_path, frame_dir)

    num_frame = len(frame_paths)  #视频秒数*FPS
    assert num_frame != 0
    print("Frame Number：", num_frame)

    # 帧图像高度和宽度
    h, w, _ = frames[0].shape

    # Get clip_len, frame_interval and calculate center index of each clip
    data_process_pipeline = build_pipeline(config.PIPELINE.test)  #测试时输出处理流水配置

    clip_len = config.PIPELINE.test.sample['clip_len']
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    frame_interval = config.PIPELINE.test.sample['frame_interval']

    # 此处关键帧每秒取一个
    clip_len = config.PIPELINE.test.sample['clip_len']
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    frame_interval = config.PIPELINE.test.sample['frame_interval']
    window_size = clip_len * frame_interval
    timestamps = np.arange(window_size // 2, (num_frame + 1 - window_size // 2),
                           args.predict_stepsize)
    print("timetamps number:", len(timestamps))

    # get selected frame list according to timestamps
    selected_frame_list = []
    for timestamp in timestamps:
        selected_frame_list.append(frame_paths[timestamp - 1])

    # Load label_map
    label_map_path = config.DATASET.test['label_file']
    categories, class_whitelist = read_labelmap(open(label_map_path))
    label_map = {}
    for item in categories:
        id = item['id']
        name = item['name']
        label_map[id] = name

    # Construct model.
    if config.MODEL.backbone.get('pretrained'):
        config.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(config.MODEL)

    model.eval()
    state_dicts = load(args.weights)
    model.set_state_dict(state_dicts)

    detection_result_dir = 'tmp_detection'
    detection_model_name = args.detection_model_name
    detection_model_weights = args.detection_model_weights
    detection_txt_list = detection_inference(selected_frame_list,
                                             detection_result_dir,
                                             detection_model_name,
                                             detection_model_weights)
    assert len(detection_txt_list) == len(timestamps)

    print('Performing SpatioTemporal Action Detection for each clip')
    human_detections = []
    predictions = []

    index = 0
    for timestamp, detection_txt_path in zip(timestamps, detection_txt_list):
        proposals, scores = get_detection_result(
            detection_txt_path, h, w,
            (float)(config.DATASET.test['person_det_score_thr']))
        if proposals.shape[0] == 0:
            predictions.append(None)
            human_detections.append(None)
            continue

        human_detections.append(proposals)

        result = get_timestep_result(frame_dir,
                                     timestamp,
                                     clip_len,
                                     frame_interval,
                                     FPS=FPS)
        result["proposals"] = proposals
        result["scores"] = scores

        new_result = data_process_pipeline(result)
        proposals = new_result['proposals']

        img_slow = new_result['imgs'][0]
        img_slow = img_slow[np.newaxis, :]
        img_fast = new_result['imgs'][1]
        img_fast = img_fast[np.newaxis, :]

        proposals = proposals[np.newaxis, :]

        scores = scores[np.newaxis, :]

        img_shape = np.asarray(new_result['img_shape'])
        img_shape = img_shape[np.newaxis, :]

        data = [
            paddle.to_tensor(img_slow, dtype='float32'),
            paddle.to_tensor(img_fast, dtype='float32'),
            paddle.to_tensor(proposals, dtype='float32'), scores,
            paddle.to_tensor(img_shape, dtype='int32')
        ]

        with paddle.no_grad():
            result = model(data, mode='infer')

            result = result[0]
            prediction = []

            person_num = proposals.shape[1]
            # N proposals
            for i in range(person_num):
                prediction.append([])

            # Perform action score thr
            for i in range(len(result)):
                if i + 1 not in class_whitelist:
                    continue
                for j in range(person_num):
                    if result[i][j, 4] > config.MODEL.head['action_thr']:
                        prediction[j].append((label_map[i + 1], result[i][j,
                                                                          4]))
            predictions.append(prediction)

        index = index + 1
        if index % 10 == 0:
            print(index, "/", len(timestamps))

    results = []
    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int)

    dense_n = int(args.predict_stepsize / args.output_stepsize)  #30
    frames = [
        cv2.imread(frame_paths[i - 1])
        for i in dense_timestamps(timestamps, dense_n)
    ]

    vis_frames = visualize(frames, results)

    try:
        import moviepy.editor as mpy
    except ImportError:
        raise ImportError('Please install moviepy to enable output file')

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=args.output_fps)
    vid.write_videofile(args.out_filename)
    print("finish write !")

    # delete tmp files and dirs
    shutil.rmtree(frame_dir)
    shutil.rmtree(detection_result_dir)


if __name__ == '__main__':
    args = parse_args()  #解析参数
    main(args)
