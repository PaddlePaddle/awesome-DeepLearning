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

import os.path as osp
import copy
import random
import numpy as np
import lmdb
import pickle

from paddlenlp.transformers import BertTokenizer
from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class MSRVTTDataset(BaseDataset):
    """MSR-VTT dataset for text-video clip retrieval.
    """
    def __init__(
        self,
        file_path,
        pipeline,
        features_path,
        bert_model="bert-base-uncased",
        padding_index=0,
        max_seq_length=36,
        max_region_num=36,
        max_action_num=5,
        vision_feature_dim=2048,
        action_feature_dim=2048,
        spatials_dim=5,
        data_prefix=None,
        test_mode=False,
    ):
        self.features_path = features_path
        self.bert_model = bert_model
        self.padding_index = padding_index
        self.max_seq_length = max_seq_length
        self.max_region_num = max_region_num
        self._max_action_num = max_action_num
        self.vision_feature_dim = vision_feature_dim
        self.action_feature_dim = action_feature_dim
        self.spatials_dim = spatials_dim
        self._tokenizer = BertTokenizer.from_pretrained(bert_model,
                                                        do_lower_case=True)
        super().__init__(file_path, pipeline, data_prefix, test_mode)
        self.tokenize()
        self.gen_feature()

    def load_file(self):
        """Load index file to get video information."""
        with open(self.file_path) as fin:
            self.image_entries = []
            self.caption_entries = []
            for line in fin.readlines():
                line = line.strip()
                vid_id = line.split(',')[0]
                self.image_entries.append(vid_id)
                self.caption_entries.append({
                    "caption": line.split(',')[1],
                    "vid_id": vid_id
                })
        self.env = lmdb.open(self.features_path)

    def tokenize(self):
        for entry in self.caption_entries:
            tokens = []
            tokens.append("[CLS]")
            for token in self._tokenizer.tokenize(entry["caption"]):
                tokens.append(token)
            tokens.append("[SEP]")
            tokens = self._tokenizer.convert_tokens_to_ids(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self.max_seq_length:
                padding = [self.padding_index
                           ] * (self.max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            entry["token"] = np.array(tokens).astype('int64')
            entry["input_mask"] = np.array(input_mask)
            entry["segment_ids"] = np.array(segment_ids).astype('int64')

    def get_image_feature(self, video_id):
        video_id = str(video_id).encode()
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(video_id))
            video_id = item["video_id"]
            image_h = int(item["image_h"])
            image_w = int(item["image_w"])

            features = item["features"].reshape(-1, self.vision_feature_dim)
            boxes = item["boxes"].reshape(-1, 4)

            num_boxes = features.shape[0]
            g_feat = np.sum(features, axis=0) / num_boxes
            num_boxes = num_boxes + 1
            features = np.concatenate(
                [np.expand_dims(g_feat, axis=0), features], axis=0)

            action_features = item["action_features"].reshape(
                -1, self.action_feature_dim)

            image_location = np.zeros((boxes.shape[0], self.spatials_dim),
                                      dtype=np.float32)
            image_location[:, :4] = boxes
            image_location[:,
                           4] = ((image_location[:, 3] - image_location[:, 1]) *
                                 (image_location[:, 2] - image_location[:, 0]) /
                                 (float(image_w) * float(image_h)))

            image_location[:, 0] = image_location[:, 0] / float(image_w)
            image_location[:, 1] = image_location[:, 1] / float(image_h)
            image_location[:, 2] = image_location[:, 2] / float(image_w)
            image_location[:, 3] = image_location[:, 3] / float(image_h)

            g_location = np.array([0, 0, 1, 1, 1])
            image_location = np.concatenate(
                [np.expand_dims(g_location, axis=0), image_location], axis=0)
        return features, num_boxes, image_location, action_features

    def gen_feature(self):
        num_inst = len(self.image_entries)  #1000
        self.features_all = np.zeros(
            (num_inst, self.max_region_num, self.vision_feature_dim))
        self.action_features_all = np.zeros(
            (num_inst, self._max_action_num, self.action_feature_dim))
        self.spatials_all = np.zeros(
            (num_inst, self.max_region_num, self.spatials_dim))
        self.image_mask_all = np.zeros((num_inst, self.max_region_num))
        self.action_mask_all = np.zeros((num_inst, self._max_action_num))

        for i, image_id in enumerate(self.image_entries):
            features, num_boxes, boxes, action_features = self.get_image_feature(
                image_id)

            mix_num_boxes = min(int(num_boxes), self.max_region_num)
            mix_boxes_pad = np.zeros((self.max_region_num, self.spatials_dim))
            mix_features_pad = np.zeros(
                (self.max_region_num, self.vision_feature_dim))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self.max_region_num:
                image_mask.append(0)
            action_mask = [1] * (self._max_action_num)
            while len(action_mask) < self._max_action_num:
                action_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            x = action_features.shape[0]
            self.action_features_all[i][:x] = action_features[:]
            self.image_mask_all[i] = np.array(image_mask)
            self.action_mask_all[i] = np.array(action_mask)
            self.spatials_all[i] = mix_boxes_pad

        self.features_all = self.features_all.astype("float32")
        self.action_features_all = self.action_features_all.astype("float32")
        self.image_mask_all = self.image_mask_all.astype("int64")
        self.action_mask_all = self.action_mask_all.astype("int64")
        self.spatials_all = self.spatials_all.astype("float32")

    def prepare_train(self, idx):
        pass

    def prepare_test(self, idx):
        entry = self.caption_entries[idx]
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        target_all = np.zeros(1000)
        for i, image_id in enumerate(self.image_entries):
            if image_id == entry["vid_id"]:
                target_all[i] = 1

        return (
            caption,
            self.action_features_all,
            self.features_all,
            self.spatials_all,
            segment_ids,
            input_mask,
            self.image_mask_all,
            self.action_mask_all,
            target_all,
        )

    def __len__(self):
        return len(self.caption_entries)
