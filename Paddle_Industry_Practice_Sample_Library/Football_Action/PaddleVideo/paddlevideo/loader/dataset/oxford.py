# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import copy
from os import path as osp

from PIL import Image

from ..registry import DATASETS
from .base import BaseDataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


@DATASETS.register()
class MonoDataset(BaseDataset):
    def __init__(self,
                 file_path,
                 data_prefix,
                 pipeline,
                 num_retries=0,
                 suffix='.png',
                 **kwargs):
        self.num_retries = num_retries
        self.suffix = suffix
        super().__init__(file_path, pipeline, data_prefix, **kwargs)

    def load_file(self):
        info = []
        with open(self.file_path, 'r') as f:
            for line in f:
                filename = line.strip() + self.suffix
                folder = osp.dirname(filename)
                frame_index = line.strip().split('/')[1]
                info.append(
                    dict(data_path=self.data_prefix,
                         filename=filename,
                         folder=folder,
                         frame_index=int(frame_index)))
        return info

    def prepare_train(self, idx):
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        results['imgs']['idx'] = idx
        return results['imgs'], results['day_or_night']

    def prepare_test(self, idx):
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        return results['imgs'], results['day_or_night']
