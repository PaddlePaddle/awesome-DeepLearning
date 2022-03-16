# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import shutil
import sys

import cv2
import imageio
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import paddle
import paddle.nn.functional as F
import pandas
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from abc import abstractmethod

from paddlevideo.loader.builder import build_pipeline
from paddlevideo.loader.pipelines import (
    AutoPadding, CenterCrop, DecodeSampler, FeatureDecoder, FrameDecoder,
    GroupResize, Image2Array, ImageDecoder, JitterScale, MultiCrop,
    Normalization, PackOutput, Sampler, SamplerPkl, Scale, SkeletonNorm,
    TenCrop, ToArray, UniformCrop, VideoDecoder)
from paddlevideo.metrics.ava_utils import read_labelmap
from paddlevideo.metrics.bmn_metric import boundary_choose, soft_nms
from paddlevideo.utils import Registry, build, get_config

from ava_predict import (detection_inference, frame_extraction,
                         get_detection_result, get_timestep_result, pack_result,
                         visualize)

INFERENCE = Registry('inference')


def decode(filepath, args):
    num_seg = args.num_seg
    seg_len = args.seg_len

    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    average_dur = int(len(sampledFrames) / num_seg)
    imgs = []
    for i in range(num_seg):
        idx = 0
        if average_dur >= seg_len:
            idx = (average_dur - 1) // 2
            idx += i * average_dur
        elif average_dur >= 1:
            idx += i * average_dur
        else:
            idx = i

        for jj in range(idx, idx + seg_len):
            imgbuf = sampledFrames[int(jj % len(sampledFrames))]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs


def preprocess(img, args):
    img = {"imgs": img}
    resize_op = Scale(short_size=args.short_size)
    img = resize_op(img)
    ccrop_op = CenterCrop(target_size=args.target_size)
    img = ccrop_op(img)
    to_array = Image2Array()
    img = to_array(img)
    if args.normalize:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        normalize_op = Normalization(mean=img_mean, std=img_std)
        img = normalize_op(img)
    return img['imgs']


def postprocess(output, args):
    output = output.flatten()
    output = F.softmax(paddle.to_tensor(output)).numpy()
    classes = np.argpartition(output, -args.top_k)[-args.top_k:]
    classes = classes[np.argsort(-output[classes])]
    scores = output[classes]
    return classes, scores


def build_inference_helper(cfg):
    return build(cfg, INFERENCE)


class Base_Inference_helper():
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

    @abstractmethod
    def preprocess(self, input_file):
        pass

    def preprocess_batch(self, file_list):
        batched_inputs = []
        for file in file_list:
            inputs = self.preprocess(file)
            batched_inputs.append(inputs)
        batched_inputs = [
            np.concatenate([item[i] for item in batched_inputs])
            for i in range(len(batched_inputs[0]))
        ]
        self.input_file = file_list
        return batched_inputs

    def postprocess(self, output, print_output=True):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]
        N = len(self.input_file)
        if output.shape[0] != N:
            output = output.reshape([N] + [output.shape[0] // N] +
                                    list(output.shape[1:]))  # [N, T, C]
            output = output.mean(axis=1)  # [N, C]
        output = F.softmax(paddle.to_tensor(output), axis=-1).numpy()
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                for j in range(self.top_k):
                    print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                    print("\ttop-{0} score: {1}".format(j + 1, scores[j]))


@INFERENCE.register()
class ppTSM_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            VideoDecoder(),
            Sampler(self.num_seg, self.seg_len, valid_mode=True),
            Scale(self.short_size),
            CenterCrop(self.target_size),
            Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]


@INFERENCE.register()
class ppTSN_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=25,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            VideoDecoder(),
            Sampler(self.num_seg,
                    self.seg_len,
                    valid_mode=True,
                    select_left=True),
            Scale(self.short_size,
                  fixed_ratio=True,
                  do_round=True,
                  backend='cv2'),
            TenCrop(self.target_size),
            Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]


@INFERENCE.register()
class BMN_Inference_helper(Base_Inference_helper):
    def __init__(self, feat_dim, dscale, tscale, result_path):
        self.feat_dim = feat_dim
        self.dscale = dscale
        self.tscale = tscale
        self.result_path = result_path
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        file_info = json.load(open(input_file))
        self.feat_path = file_info['feat_path']
        self.video_duration = file_info['duration_second']
        feat = np.load(self.feat_path).astype('float32').T
        res = np.expand_dims(feat, axis=0).copy()

        return [res]

    def postprocess(self, outputs, print_output=True):
        """
        output: list
        """
        pred_bm, pred_start, pred_end = outputs
        self._gen_props(pred_bm, pred_start[0], pred_end[0], print_output)

    def _gen_props(self, pred_bm, pred_start, pred_end, print_output):
        snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]

        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        start_mask = boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_vector_list = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < self.tscale and start_mask[
                        start_index] == 1 and end_mask[end_index] == 1:
                    xmin = snippet_xmins[start_index]
                    xmax = snippet_xmaxs[end_index]
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bm_score = pred_bm[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        cols = ["xmin", "xmax", "score"]
        score_vector_list = np.stack(score_vector_list)
        df = pandas.DataFrame(score_vector_list, columns=cols)

        result_dict = {}
        proposal_list = []
        df = soft_nms(df, alpha=0.4, t1=0.55, t2=0.9)
        for idx in range(min(100, len(df))):
            tmp_prop={"score":df.score.values[idx], \
                      "segment":[max(0,df.xmin.values[idx])*self.video_duration, \
                                 min(1,df.xmax.values[idx])*self.video_duration]}
            proposal_list.append(tmp_prop)

        result_dict[self.feat_path] = proposal_list

        # print top-5 predictions
        if print_output:
            print("Current video file: {0} :".format(self.feat_path))
            for pred in proposal_list[:5]:
                print(pred)

        # save result
        outfile = open(
            os.path.join(self.result_path, "bmn_results_inference.json"), "w")

        json.dump(result_dict, outfile)


@INFERENCE.register()
class TimeSformer_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=224,
                 target_size=224,
                 top_k=1,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225]):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k
        self.mean = mean
        self.std = std

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        ops = [
            VideoDecoder(backend='pyav', mode='test', num_seg=self.num_seg),
            Sampler(self.num_seg,
                    self.seg_len,
                    valid_mode=True,
                    linspace_sample=True),
            Normalization(self.mean, self.std, tensor_shape=[1, 1, 1, 3]),
            Image2Array(data_format='cthw'),
            JitterScale(self.short_size, self.short_size),
            UniformCrop(self.target_size)
        ]
        for op in ops:
            results = op(results)

        # [N,C,Tx3,H,W]
        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]


@INFERENCE.register()
class VideoSwin_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=4,
                 seg_len=32,
                 frame_interval=2,
                 short_size=224,
                 target_size=224,
                 top_k=1,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):

        self.num_seg = num_seg
        self.seg_len = seg_len
        self.frame_interval = frame_interval
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k
        self.mean = mean
        self.std = std

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        self.input_file = input_file
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        ops = [
            VideoDecoder(backend='decord', mode='valid'),
            Sampler(num_seg=self.num_seg,
                    frame_interval=self.frame_interval,
                    seg_len=self.seg_len,
                    valid_mode=True,
                    use_pil=False),
            Scale(short_size=self.short_size,
                  fixed_ratio=False,
                  keep_ratio=True,
                  backend='cv2',
                  do_round=True),
            CenterCrop(target_size=224, backend='cv2'),
            Normalization(mean=self.mean,
                          std=self.std,
                          tensor_shape=[3, 1, 1, 1],
                          inplace=True),
            Image2Array(data_format='cthw')
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]

    def postprocess(self, output, print_output=True):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]
        N = len(self.input_file)
        if output.shape[0] != N:
            output = output.reshape([N] + [output.shape[0] // N] +
                                    list(output.shape[1:]))  # [N, T, C]
            output = output.mean(axis=1)  # [N, C]
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                for j in range(self.top_k):
                    print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                    print("\ttop-{0} score: {1}".format(j + 1, scores[j]))


@INFERENCE.register()
class VideoSwin_TableTennis_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=1,
                 seg_len=32,
                 short_size=256,
                 target_size=224,
                 top_k=1):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'frame_dir': input_file, 'suffix': 'img_{:05}.jpg'}
        img_mean = [123.675, 116.28, 103.53]
        img_std = [58.395, 57.12, 57.375]
        ops = [
            FrameDecoder(),
            SamplerPkl(num_seg=self.num_seg,
                       seg_len=self.seg_len,
                       backend='cv2',
                       valid_mode=True),
            Scale(short_size=self.short_size,
                  fixed_ratio=False,
                  keep_ratio=True,
                  backend='cv2',
                  do_round=True),
            UniformCrop(target_size=self.target_size, backend='cv2'),
            Normalization(mean=img_mean,
                          std=img_std,
                          tensor_shape=[3, 1, 1, 1],
                          inplace=True),
            Image2Array(data_format='cthw')
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]

    def add_text_to_video(
            self,
            video_path,
            output_dir="applications/TableTennis/ActionRecognition/results",
            text=None):
        os.makedirs(output_dir, exist_ok=True)
        if video_path.endswith('.pkl'):
            try:
                import cPickle as pickle
                from cStringIO import StringIO
            except ImportError:
                import pickle
                from io import BytesIO
            from PIL import Image
            data_loaded = pickle.load(open(video_path, 'rb'), encoding='bytes')
            _, _, frames = data_loaded
            frames_len = len(frames)

        else:
            videoCapture = cv2.VideoCapture()
            videoCapture.open(video_path)

            fps = videoCapture.get(cv2.CAP_PROP_FPS)
            frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames_len = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
            print("fps=", int(fps), "frames=", int(frames_len), "scale=",
                  f"{frame_height}x{frame_width}")

        frames_rgb_list = []
        for i in range(int(frames_len)):
            if video_path.endswith('.pkl'):
                frame = np.array(
                    Image.open(BytesIO(frames[i])).convert("RGB").resize(
                        (240, 135)))[:, :, ::-1].astype('uint8')
            else:
                _, frame = videoCapture.read()
            frame = cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                                1.0, (0, 0, 255), 2)
            frames_rgb_list.append(frame[:, :, ::-1])  # bgr to rgb
        if not video_path.endswith('.pkl'):
            videoCapture.release()
        cv2.destroyAllWindows()
        output_filename = os.path.basename(video_path)
        output_filename = output_filename.split('.')[0] + '.gif'
        imageio.mimsave(f'{output_dir}/{output_filename}',
                        frames_rgb_list,
                        'GIF',
                        duration=0.00085)

    def postprocess(self, output, print_output=True, save_gif=True):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]
        N = len(self.input_file)
        if output.shape[0] != N:
            output = output.reshape([N] + [output.shape[0] // N] +
                                    list(output.shape[1:]))  # [N, T, C]
            output = output.mean(axis=1)  # [N, C]
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                for j in range(self.top_k):
                    print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                    print("\ttop-{0} score: {1}".format(j + 1, scores[j]))
            if save_gif:
                self.add_text_to_video(
                    self.input_file[0],
                    text=f"{str(classes[0])} {float(scores[0]):.5f}")


@INFERENCE.register()
class SlowFast_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_frames=32,
                 sampling_rate=2,
                 target_size=256,
                 alpha=8,
                 top_k=1):
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.target_size = target_size
        self.alpha = alpha
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {
            'filename': input_file,
            'temporal_sample_index': 0,
            'spatial_sample_index': 0,
            'temporal_num_clips': 1,
            'spatial_num_clips': 1
        }
        img_mean = [0.45, 0.45, 0.45]
        img_std = [0.225, 0.225, 0.225]
        ops = [
            DecodeSampler(self.num_frames, self.sampling_rate, test_mode=True),
            JitterScale(self.target_size, self.target_size),
            MultiCrop(self.target_size),
            Image2Array(transpose=False),
            Normalization(img_mean, img_std, tensor_shape=[1, 1, 1, 3]),
            PackOutput(self.alpha),
        ]
        for op in ops:
            results = op(results)

        res = []
        for item in results['imgs']:
            res.append(np.expand_dims(item, axis=0).copy())
        return res

    def postprocess(self, output, print_output=True):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]

        N = len(self.input_file)
        if output.shape[0] != N:
            output = output.reshape([N] + [output.shape[0] // N] +
                                    list(output.shape[1:]))  # [N, T, C]
            output = output.mean(axis=1)  # [N, C]
        # output = F.softmax(paddle.to_tensor(output), axis=-1).numpy() # done in it's head
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                for j in range(self.top_k):
                    print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                    print("\ttop-{0} score: {1}".format(j + 1, scores[j]))


@INFERENCE.register()
class STGCN_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_channels,
                 window_size,
                 vertex_nums,
                 person_nums,
                 top_k=1):
        self.num_channels = num_channels
        self.window_size = window_size
        self.vertex_nums = vertex_nums
        self.person_nums = person_nums
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        data = np.load(input_file)
        results = {'data': data}
        ops = [AutoPadding(window_size=self.window_size), SkeletonNorm()]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['data'], axis=0).copy()
        return [res]


@INFERENCE.register()
class AttentionLSTM_Inference_helper(Base_Inference_helper):
    def __init__(
            self,
            num_classes,  #Optional, the number of classes to be classified.
            feature_num,
            feature_dims,
            embedding_size,
            lstm_size,
            top_k=1):
        self.num_classes = num_classes
        self.feature_num = feature_num
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        ops = [FeatureDecoder(num_classes=self.num_classes, has_label=False)]
        for op in ops:
            results = op(results)

        res = []
        for modality in ['rgb', 'audio']:
            res.append(
                np.expand_dims(results[f'{modality}_data'], axis=0).copy())
            res.append(
                np.expand_dims(results[f'{modality}_len'], axis=0).copy())
            res.append(
                np.expand_dims(results[f'{modality}_mask'], axis=0).copy())
        return res


@INFERENCE.register()
class TransNetV2_Inference_helper():
    def __init__(self,
                 num_frames,
                 height,
                 width,
                 num_channels,
                 threshold=0.5,
                 output_path=None,
                 visualize=True):
        self._input_size = (height, width, num_channels)
        self.output_path = output_path
        self.len_frames = 0
        self.threshold = threshold
        self.visualize = visualize

    def input_iterator(self, frames):
        # return windows of size 100 where the first/last 25 frames are from the previous/next batch
        # the first and last window must be padded by copies of the first and last frame of the video
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (
            len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate([start_frame] * no_padded_frames_start +
                                       [frames] +
                                       [end_frame] * no_padded_frames_end, 0)

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            out = out.astype(np.float32)
            ptr += 50
            yield out[np.newaxis]

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: iterator
        """
        import ffmpeg
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        self.input_file = input_file
        self.filename = os.path.splitext(os.path.split(self.input_file)[1])[0]
        video_stream, err = ffmpeg.input(
            self.input_file).output("pipe:",
                                    format="rawvideo",
                                    pix_fmt="rgb24",
                                    s="48x27").run(capture_stdout=True,
                                                   capture_stderr=True)
        self.frames = np.frombuffer(video_stream,
                                    np.uint8).reshape([-1, 27, 48, 3])
        self.len_frames = len(self.frames)

        return self.input_iterator(self.frames)

    def predictions_to_scenes(self, predictions):
        predictions = (predictions > self.threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def visualize_predictions(self, frames, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        # pad frames so that length of the video is divisible by width
        # pad frames also by len(predictions) pixels in width in order to show predictions
        pad_with = width - len(frames) % width if len(
            frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)),
                                 (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width),
                             axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # iterate over all frames
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            # we can visualize multiple predictions per single frame
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value),
                              fill=tuple(color),
                              width=1)
        return img

    def postprocess(self, outputs, print_output=True):
        """
        output: list
        """
        predictions = []
        for output in outputs:
            single_frame_logits, all_frames_logits = output
            single_frame_pred = F.sigmoid(paddle.to_tensor(single_frame_logits))
            all_frames_pred = F.sigmoid(paddle.to_tensor(all_frames_logits))
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                                all_frames_pred.numpy()[0, 25:75, 0]))
        single_frame_pred = np.concatenate(
            [single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate(
            [all_ for single_, all_ in predictions])
        single_frame_predictions, all_frame_predictions = single_frame_pred[:
                                                                            self
                                                                            .
                                                                            len_frames], all_frames_pred[:
                                                                                                         self
                                                                                                         .
                                                                                                         len_frames]

        scenes = self.predictions_to_scenes(single_frame_predictions)

        if print_output:
            print("Current video file: {0}".format(self.input_file))
            print("\tShot Boundarys: {0}".format(scenes))

        if self.output_path:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            predictions = np.stack(
                [single_frame_predictions, all_frame_predictions], 1)
            predictions_file = os.path.join(self.output_path,
                                            self.filename + "_predictions.txt")
            np.savetxt(predictions_file, predictions, fmt="%.6f")
            scenes_file = os.path.join(self.output_path,
                                       self.filename + "_scenes.txt")
            np.savetxt(scenes_file, scenes, fmt="%d")

            if self.visualize:
                pil_image = self.visualize_predictions(
                    self.frames,
                    predictions=(single_frame_predictions,
                                 all_frame_predictions))
                image_file = os.path.join(self.output_path,
                                          self.filename + "_vis.png")
                pil_image.save(image_file)


@INFERENCE.register()
class ADDS_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 frame_idxs=[0],
                 num_scales=4,
                 side_map={
                     "2": 2,
                     "3": 3,
                     "l": 2,
                     "r": 3
                 },
                 height=256,
                 width=512,
                 full_res_shape=None,
                 num_channels=None,
                 img_ext=".png",
                 K=None):

        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.side_map = side_map
        self.full_res_shape = full_res_shape
        self.img_ext = img_ext
        self.height = height
        self.width = width
        self.K = K

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {
            'filename': input_file,
            'mode': 'infer',
            'day_or_night': 'day',
        }
        ops = [
            ImageDecoder(
                backend='pil',
                dataset='kitti',
                frame_idxs=self.frame_idxs,
                num_scales=self.num_scales,
                side_map=self.side_map,
                full_res_shape=self.full_res_shape,
                img_ext=self.img_ext,
            ),
            GroupResize(
                height=self.height,
                width=self.width,
                K=self.K,
                scale=1,
                mode='infer',
            ),
            ToArray(),
        ]
        for op in ops:
            results = op(results)
        res = results['imgs'][('color', 0, 0)]
        res = np.expand_dims(res, axis=0).copy()
        return [res]

    def postprocess(self, output, print_output, save_dir='data/'):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        print(len(output))
        N = len(self.input_file)
        for i in range(N):
            pred_depth = output[i]  # [H, W]
            if print_output:
                print("Current input image: {0}".format(self.input_file[i]))
                file_name = os.path.basename(self.input_file[i]).split('.')[0]
                save_path = os.path.join(save_dir,
                                         file_name + "_depth" + ".png")
                pred_depth_color = self._convertPNG(pred_depth)
                pred_depth_color.save(save_path)
                print(f"pred depth image saved to: {save_path}")

    def _convertPNG(self, image_numpy):
        disp_resized = cv2.resize(image_numpy, (1280, 640))
        disp_resized_np = disp_resized
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] *
                          255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        return im


@INFERENCE.register()
class AVA_SlowFast_FastRCNN_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 detection_model_name,
                 detection_model_weights,
                 config_file_path,
                 predict_stepsize=8,
                 output_stepsize=4,
                 output_fps=6,
                 out_filename='ava_det_demo.mp4',
                 num_frames=32,
                 alpha=4,
                 target_size=256):
        self.detection_model_name = detection_model_name
        self.detection_model_weights = detection_model_weights

        self.config = get_config(config_file_path,
                                 show=False)  #parse config file
        self.predict_stepsize = predict_stepsize
        self.output_stepsize = output_stepsize
        self.output_fps = output_fps
        self.out_filename = out_filename
        self.num_frames = num_frames
        self.alpha = alpha
        self.target_size = target_size

    def preprocess(self, input_file):
        """
        input_file: str, file path
        """

        frame_dir = 'tmp_frames'
        self.frame_paths, frames, FPS = frame_extraction(input_file, frame_dir)
        num_frame = len(self.frame_paths)  #视频秒数*FPS
        assert num_frame != 0

        # 帧图像高度和宽度
        h, w, _ = frames[0].shape

        # Get clip_len, frame_interval and calculate center index of each clip
        data_process_pipeline = build_pipeline(
            self.config.PIPELINE.test)  #测试时输出处理流水配置

        clip_len = self.config.PIPELINE.test.sample['clip_len']
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'
        frame_interval = self.config.PIPELINE.test.sample['frame_interval']

        # 此处关键帧每秒取一个
        clip_len = self.config.PIPELINE.test.sample['clip_len']
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'
        frame_interval = self.config.PIPELINE.test.sample['frame_interval']
        window_size = clip_len * frame_interval
        timestamps = np.arange(window_size // 2,
                               (num_frame + 1 - window_size // 2),
                               self.predict_stepsize)

        selected_frame_list = []
        for timestamp in timestamps:
            selected_frame_list.append(self.frame_paths[timestamp - 1])

        # Load label_map
        label_map_path = self.config.DATASET.test['label_file']
        self.categories, self.class_whitelist = read_labelmap(
            open(label_map_path))
        label_map = {}
        for item in self.categories:
            id = item['id']
            name = item['name']
            label_map[id] = name

        self.label_map = label_map

        detection_result_dir = 'tmp_detection'
        detection_model_name = self.detection_model_name
        detection_model_weights = self.detection_model_weights
        detection_txt_list = detection_inference(selected_frame_list,
                                                 detection_result_dir,
                                                 detection_model_name,
                                                 detection_model_weights)
        assert len(detection_txt_list) == len(timestamps)

        human_detections = []
        data_list = []
        person_num_list = []

        for timestamp, detection_txt_path in zip(timestamps,
                                                 detection_txt_list):
            proposals, scores = get_detection_result(
                detection_txt_path, h, w,
                (float)(self.config.DATASET.test['person_det_score_thr']))

            if proposals.shape[0] == 0:
                #person_num_list.append(0)
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
                paddle.to_tensor(proposals, dtype='float32'),
                paddle.to_tensor(img_shape, dtype='int32')
            ]

            person_num = proposals.shape[1]
            person_num_list.append(person_num)

            data_list.append(data)

        self.human_detections = human_detections
        self.person_num_list = person_num_list
        self.timestamps = timestamps
        self.frame_dir = frame_dir
        self.detection_result_dir = detection_result_dir

        return data_list

    def postprocess(self, outputs, print_output=True):
        """
        output: list
        """
        predictions = []

        assert len(self.person_num_list) == len(outputs)

        #print("***  self.human_detections",len( self.human_detections))
        #print("***  outputs",len( outputs))

        index = 0
        for t_index in range(len(self.timestamps)):
            if self.human_detections[t_index] is None:
                predictions.append(None)
                continue

            human_detection = self.human_detections[t_index]

            output = outputs[index]
            result = output  #长度为类别个数，不包含背景

            person_num = self.person_num_list[index]

            index = index + 1

            prediction = []

            if human_detection is None:
                predictions.append(None)
                continue

            # N proposals
            for i in range(person_num):
                prediction.append([])

            # Perform action score thr
            for i in range(len(result)):  # for class
                if i + 1 not in self.class_whitelist:
                    continue
                for j in range(person_num):
                    if result[i][j, 4] > self.config.MODEL.head['action_thr']:
                        prediction[j].append(
                            (self.label_map[i + 1], result[i][j, 4]
                             ))  # label_map is a dict, label index start from 1
            predictions.append(prediction)

        results = []
        for human_detection, prediction in zip(self.human_detections,
                                               predictions):
            results.append(pack_result(human_detection, prediction))

        def dense_timestamps(timestamps, n):
            """Make it nx frames."""
            old_frame_interval = (timestamps[1] - timestamps[0])
            start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
            new_frame_inds = np.arange(
                len(timestamps) * n) * old_frame_interval / n + start
            return new_frame_inds.astype(np.int)

        dense_n = int(self.predict_stepsize / self.output_stepsize)  #30
        frames = [
            cv2.imread(self.frame_paths[i - 1])
            for i in dense_timestamps(self.timestamps, dense_n)
        ]

        vis_frames = visualize(frames, results)

        try:
            import moviepy.editor as mpy
        except ImportError:
            raise ImportError('Please install moviepy to enable output file')

        vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                    fps=self.output_fps)
        vid.write_videofile(self.out_filename)
        print("finish write !")

        # delete tmp files and dirs
        shutil.rmtree(self.frame_dir)
        shutil.rmtree(self.detection_result_dir)
