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

import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))


import numpy as np
import tarfile
import requests
from tqdm import tqdm
from tools import utils
import shutil

from paddle.inference import Config
from paddle.inference import create_predictor

__all__ = ['PaddleVideo']
BASE_DIR = os.path.expanduser("~/.paddlevideo_inference/")
BASE_INFERENCE_MODEL_DIR = os.path.join(BASE_DIR, 'inference_model')
BASE_VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')

model_names = {'ppTSM','TSM','TSN'}


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=Config.Precision.Half
            if args.use_fp16 else Config.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor

def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        raise Exception("Something went wrong while downloading models")

def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = [
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
    ]
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    ) or not os.path.exists(
        os.path.join(model_storage_directory, 'inference.pdmodel')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path) #download

        #save to directory
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)

def load_label_name_dict(path):
    result = {}
    if not os.path.exists(path):
        print(
            'Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!'
        )
    else:
        for line in open(path, 'r'):
            partition = line.split('\n')[0].partition(' ')
            try:
                result[int(partition[0])] = str(partition[-1])
            except:
                result = {}
                break
    return result

def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain == True:

        # general params
        parser = argparse.ArgumentParser(add_help=add_help)
        parser.add_argument("--model_name", type=str,default='')
        parser.add_argument("-v", "--video_file", type=str,default='')
        parser.add_argument("--use_gpu", type=str2bool, default=True)

        # params for decode and sample
        parser.add_argument("--num_seg", type=int, default=8)
        parser.add_argument("--seg_len", type=int, default=1)

        # params for preprocess
        parser.add_argument("--short_size", type=int, default=256)
        parser.add_argument("--target_size", type=int, default=224)
        parser.add_argument("--normalize", type=str2bool, default=True)

        # params for predict
        parser.add_argument("--model_file", type=str,default='')
        parser.add_argument("--params_file", type=str)
        parser.add_argument("-b", "--batch_size", type=int, default=1)
        parser.add_argument("--use_fp16", type=str2bool, default=False)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)
        parser.add_argument("--top_k", type=int, default=1)
        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--label_name_path",type=str,default='')

        return parser.parse_args()

    else:
        return argparse.Namespace(
            model_name='',
            video_file='',
            use_gpu=False,
            num_seg=8,
            seg_len=1,
            short_size=256,
            target_size=224,
            normalize=True,
            model_file='',
            params_file='',
            batch_size=1,
            use_fp16=False,
            ir_optim=True,
            use_tensorrt=False,
            gpu_mem=8000,
            top_k=1,
            enable_mkldnn=False,
            label_name_path='')

def get_video_list(video_file):
    videos_lists = []
    if video_file is None or not os.path.exists(video_file):
        raise Exception("not found any video file in {}".format(video_file))

    video_end = ['mp4','avi']
    if os.path.isfile(video_file) and video_file.split('.')[-1] in video_end:
        videos_lists.append(video_file)
    elif os.path.isdir(video_file):
        for single_file in os.listdir(video_file):
            if single_file.split('.')[-1] in video_end:
                videos_lists.append(os.path.join(video_file, single_file))
    if len(videos_lists) == 0:
        raise Exception("not found any video file in {}".format(video_file))
    return videos_lists

class PaddleVideo(object):
    print('Inference models that Paddle provides are listed as follows:\n\n{}'.
          format(model_names), '\n')

    def __init__(self, **kwargs):
        process_params = parse_args(mMain=False,add_help=False)
        process_params.__dict__.update(**kwargs)

        if not os.path.exists(process_params.model_file):
            if process_params.model_name is None:
                raise Exception(
                    'Please input model name that you want to use!')
            if process_params.model_name in model_names:
                url = 'https://videotag.bj.bcebos.com/PaddleVideo/InferenceModel/{}_infer.tar'.format(process_params.model_name)
                if not os.path.exists(
                        os.path.join(BASE_INFERENCE_MODEL_DIR,
                                     process_params.model_name)):
                    os.makedirs(
                        os.path.join(BASE_INFERENCE_MODEL_DIR,
                                     process_params.model_name))
                #create pretrained model download_path
                download_path = os.path.join(BASE_INFERENCE_MODEL_DIR,
                                             process_params.model_name)
                maybe_download(model_storage_directory=download_path, url=url)
                process_params.model_file = os.path.join(download_path,
                                                         'inference.pdmodel')
                process_params.params_file = os.path.join(
                    download_path, 'inference.pdiparams')
                process_params.label_name_path = os.path.join(
                    __dir__, '../data/k400/Kinetics-400_label_list.txt')
            else:
                raise Exception(
                    'If you want to use your own model, Please input model_file as model path!'
                )
        else:
            print('Using user-specified model and params!')
        print("process params are as follows: \n{}".format(process_params))
        self.label_name_dict = load_label_name_dict(
            process_params.label_name_path)

        self.args = process_params
        self.predictor = create_paddle_predictor(process_params)

    def predict(self,video):
        """
        predict label of video with paddlevideo_clas
        Args:
            video:input video for clas, support single video , internet url, folder path containing series of videos
        Returns:
            list[dict:{videoname: "",class_ids: [], scores: [], label_names: []}],if label name path is None,label names will be empty
        """
        video_list = []
        assert isinstance(video, (str, np.ndarray))

        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])

        if isinstance(video, str):
            # download internet video,
            if video.startswith('http'):
                if not os.path.exists(BASE_VIDEOS_DIR):
                    os.makedirs(BASE_VIDEOS_DIR)
                video_path = os.path.join(BASE_VIDEOS_DIR, 'tmp.mp4')
                download_with_progressbar(video, video_path)
                print("Current using video from Internet:{}, renamed as: {}".
                      format(video, video_path))
                video = video_path
            video_list = get_video_list(video)
        else:
            if isinstance(video, np.ndarray):
                video_list = [video]
            else:
                print('Please input legal video!')

        total_result = []
        for filename in video_list:
            if isinstance(filename, str):
                v = utils.decode(filename, self.args)  
                assert v is not None, "Error in loading video: {}".format(
                    filename)
                inputs = utils.preprocess(v, self.args)
                inputs = np.expand_dims(
                    inputs, axis=0).repeat(
                    1, axis=0).copy()
            else:
                inputs = filename

            input_tensor.copy_from_cpu(inputs)

            self.predictor.run()

            outputs = output_tensor.copy_to_cpu()
            classes, scores = utils.postprocess(outputs, self.args)
            label_names = []
            if len(self.label_name_dict) != 0:
                label_names = [self.label_name_dict[c] for c in classes]
            result = {
                "videoname": filename if isinstance(filename, str) else 'video',
                "class_ids": classes.tolist(),
                "scores": scores.tolist(),
                "label_names": label_names,
            }
            total_result.append(result)
        return total_result

def main():
    # for cmd
    args = parse_args(mMain=True)
    clas_engine = PaddleVideo(**(args.__dict__))
    print('{}{}{}'.format('*' * 10, args.video_file, '*' * 10))
    result = clas_engine.predict(args.video_file)
    if result is not None:
        print(result)


if __name__ == '__main__':
    main()
