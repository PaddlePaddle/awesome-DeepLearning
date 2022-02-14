#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import PIL.Image as pil
import skimage.transform
from PIL import Image

from ..registry import PIPELINES


@PIPELINES.register()
class ImageDecoder(object):
    """Decode Image
    """
    def __init__(self,
                 dataset,
                 frame_idxs,
                 num_scales,
                 side_map,
                 full_res_shape,
                 img_ext,
                 backend='cv2'):
        self.backend = backend
        self.dataset = dataset
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.side_map = side_map
        self.full_res_shape = full_res_shape
        self.img_ext = img_ext

    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_color(self, folder, frame_index, side):
        color = self._pil_loader(
            self.get_image_path(self.dataset, folder, frame_index, side))
        return color

    def get_image_path(self, dataset, folder, frame_index, side):
        if dataset == "kitti":
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(self.data_path, folder, f_str)
        elif dataset == "kitti_odom":
            f_str = "{:06d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(self.data_path,
                                      "sequences/{:02d}".format(int(folder)),
                                      "image_{}".format(self.side_map[side]),
                                      f_str)
        elif dataset == "kitti_depth":
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                self.data_path, folder,
                "image_0{}/data".format(self.side_map[side]), f_str)

        return image_path

    def get_depth(self, dataset, folder, frame_index, side):
        if dataset == "kitii_depth":
            f_str = "{:010d}.png".format(frame_index)
            depth_path = os.path.join(
                self.data_path, folder,
                "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
                f_str)

            depth_gt = pil.open(depth_path)
            depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
            depth_gt = np.array(depth_gt).astype(np.float32) / 256

        else:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            depth_path = os.path.join(self.data_path, folder + '_gt', f_str)

            img_file = Image.open(depth_path)
            depth_png = np.array(img_file, dtype=int)
            img_file.close()
            # make sure we have a proper 16bit depth map here.. not 8bit!
            assert np.max(depth_png) > 255, \
                "np.max(depth_png)={}, path={}".format(np.max(depth_png), depth_path)

            depth_gt = depth_png.astype(np.float) / 256.

            depth_gt = depth_gt[160:960 - 160, :]

            depth_gt = skimage.transform.resize(depth_gt,
                                                self.full_res_shape[::-1],
                                                order=0,
                                                preserve_range=True,
                                                mode='constant')

        return depth_gt

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        if results.get('mode', None) == 'infer':
            imgs = {}
            imgs[("color", 0,
                  -1)] = Image.open(results["filename"]).convert("RGB")
            results['imgs'] = imgs
            return results

        self.data_path = results['data_path']
        results['backend'] = self.backend

        imgs = {}

        results['frame_idxs'] = self.frame_idxs
        results['num_scales'] = self.num_scales

        file_name = results['filename']
        folder = results['folder']
        frame_index = results['frame_index']
        line = file_name.split('/')
        istrain = folder.split('_')[1]
        if 'mode' not in results:
            results['mode'] = istrain
        results['day_or_night'] = folder.split('_')[0]

        if istrain == "train":
            if folder[0] == 'd':
                folder2 = folder + '_fake_night'
                flag = 0
            else:
                folder2 = folder + '_fake_day'
                tmp = folder
                folder = folder2
                folder2 = tmp
                flag = 1

            if len(line) == 3:
                side = line[2]
            else:
                side = None

            results['side'] = side

            for i in self.frame_idxs:

                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    imgs[("color", i,
                          -1)] = self.get_color(folder, frame_index, other_side)
                    imgs[("color_n", i,
                          -1)] = self.get_color(folder2, frame_index,
                                                other_side)
                else:
                    imgs[("color", i,
                          -1)] = self.get_color(folder, frame_index + i, side)
                    imgs[("color_n", i,
                          -1)] = self.get_color(folder2, frame_index + i, side)

            istrain = folder.split('_')[1]
            if istrain != 'train':
                if flag:
                    depth_gt = self.get_depth(folder2, frame_index, side)
                else:
                    depth_gt = self.get_depth(folder, frame_index, side)
                imgs["depth_gt"] = np.expand_dims(depth_gt, 0)
        elif istrain == 'val':
            if len(line) == 3:
                side = line[2]
            else:
                side = None

            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    imgs[("color", i,
                          -1)] = self.get_color(folder, frame_index, other_side)
                else:

                    imgs[("color", i,
                          -1)] = self.get_color(folder, frame_index + i, side)

            # adjusting intrinsics to match each scale in the pyramid

            depth_gt = self.get_depth(self.dataset, folder, frame_index, side)
            imgs["depth_gt"] = np.expand_dims(depth_gt, 0)
        results['imgs'] = imgs

        return results
