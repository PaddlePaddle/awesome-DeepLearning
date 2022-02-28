# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import os.path as osp
import copy
import random
import numpy as np
import shutil
from PIL import Image
import cv2
from paddle.io import Dataset

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


class VOS_Test(Dataset):
    """process frames in each video
    """
    def __init__(self,
                 image_root,
                 label_root,
                 seq_name,
                 images,
                 labels,
                 pipeline=None,
                 rgb=False,
                 resolution=None):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images  # image file list
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.pipeline = pipeline
        self.rgb = rgb
        self.resolution = resolution

        self.obj_nums = []
        temp_obj_num = 0
        for img_name in self.images:
            self.obj_nums.append(temp_obj_num)
            current_label_name = img_name.split('.')[0] + '.png'
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                if temp_obj_num < np.unique(
                        current_label)[-1]:  #get object number from label_id
                    temp_obj_num = np.unique(current_label)[-1]

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def read_label(self, label_name):
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        current_img = np.array(current_img)
        height, width, channels = current_img.shape
        if self.resolution is not None:
            width = int(np.ceil(float(width) * self.resolution / float(height)))
            height = int(self.resolution)

        current_label_name = img_name.split('.')[0] + '.png'
        obj_num = self.obj_nums[idx]

        if current_label_name in self.labels:
            current_label = self.read_label(current_label_name)
            current_label = np.array(current_label)
            sample = {
                'current_img': current_img,
                'current_label': current_label
            }
        else:
            sample = {
                'current_img': current_img
            }  #only the first frame contains label

        sample['meta'] = {
            'seq_name': self.seq_name,
            'frame_num': self.num_frame,
            'obj_num': obj_num,
            'current_name': img_name,
            'height': height,
            'width': width,
            'flip': False
        }
        if self.pipeline is not None:
            sample = self.pipeline(sample)
        for s in sample:
            s['current_img'] = np.array(s['current_img'])
            if 'current_label' in s.keys():
                s['current_label'] = s['current_label']
        return sample


@DATASETS.register()
class DavisDataset(BaseDataset):
    """Davis 2017 dataset.
    """
    def __init__(
        self,
        file_path,
        result_root,
        pipeline,
        data_prefix=None,
        test_mode=False,
        year=2017,
        rgb=False,
        resolution='480p',
    ):
        self.rgb = rgb
        self.result_root = result_root
        self.resolution = resolution
        self.year = year
        self.spt = 'val' if test_mode else 'train'
        super().__init__(file_path, pipeline, data_prefix, test_mode)

    def load_file(self):
        self.image_root = os.path.join(self.file_path, 'JPEGImages',
                                       self.resolution)
        self.label_root = os.path.join(self.file_path, 'Annotations',
                                       self.resolution)
        seq_names = []
        with open(
                os.path.join(self.file_path, 'ImageSets', str(self.year),
                             self.spt + '.txt')) as f:
            seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seq_names.extend(seqs_tmp)
        self.info = list(np.unique(seq_names))
        return self.info

    def prepare_test(self, idx):
        seq_name = self.info[idx]  #video name
        images = list(
            np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
        labels = [images[0].replace('jpg', 'png')]  #we have first frame target

        # copy first frame target
        if not os.path.isfile(
                os.path.join(self.result_root, seq_name, labels[0])):
            if not os.path.exists(os.path.join(self.result_root, seq_name)):
                os.makedirs(os.path.join(self.result_root, seq_name))
            source_label_path = os.path.join(self.label_root, seq_name,
                                             labels[0])
            result_label_path = os.path.join(self.result_root, seq_name,
                                             labels[0])

            shutil.copy(source_label_path, result_label_path)

        seq_dataset = VOS_Test(self.image_root,
                               self.label_root,
                               seq_name,
                               images,
                               labels,
                               self.pipeline,
                               rgb=self.rgb,
                               resolution=480)
        return seq_dataset
