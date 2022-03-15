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

import random
import math

from paddle.distributed import ParallelEnv
import paddle.distributed as dist
from paddle.fluid.dygraph import to_variable
from paddlevideo.utils import get_logger
logger = get_logger("paddlevideo")

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import tempfile
    from nvidia.dali.plugin.paddle import DALIGenericIterator
except:
    Pipeline = object
    logger.info(
        "DALI is not installed, you can improve performance if use DALI")


def get_input_data(data):
    return to_variable(data[0]['image']), to_variable(data[0]['label'])


class TSN_Dali_loader(object):
    def __init__(self, cfg):
        self.batch_size = cfg.batch_size
        self.file_path = cfg.file_path

        self.num_seg = cfg.num_seg
        self.seglen = cfg.seglen
        self.short_size = cfg.short_size
        self.target_size = cfg.target_size

        # set num_shards and shard_id when distributed training is implemented
        self.num_shards = dist.get_world_size()
        self.shard_id = ParallelEnv().local_rank
        self.dali_mean = cfg.mean * (self.num_seg * self.seglen)
        self.dali_std = cfg.std * (self.num_seg * self.seglen)

    def build_dali_reader(self):
        """
        build dali training reader
        """
        def reader_():
            with open(self.file_path) as flist:
                full_lines = [line for line in flist]
                if (not hasattr(reader_, 'seed')):
                    reader_.seed = 0
                random.Random(reader_.seed).shuffle(full_lines)
                logger.info(f"reader shuffle seed: {reader_.seed}.")
                if reader_.seed is not None:
                    reader_.seed += 1

                per_node_lines = int(
                    math.ceil(len(full_lines) * 1.0 / self.num_shards))
                total_lines = per_node_lines * self.num_shards

                # aligned full_lines so that it can evenly divisible
                full_lines += full_lines[:(total_lines - len(full_lines))]
                assert len(full_lines) == total_lines

                # trainer get own sample
                lines = full_lines[self.shard_id:total_lines:self.num_shards]
                assert len(lines) == per_node_lines

                logger.info(
                    f"shard_id: {self.shard_id}, trainer_count: {self.num_shards}"
                )
                logger.info(
                    f"read videos from {self.shard_id * per_node_lines}, "
                    f"length: {per_node_lines}, "
                    f"lines length: {len(lines)}, "
                    f"total: {len(full_lines)}")

            video_files = ''
            for item in lines:
                video_files += item
            tf = tempfile.NamedTemporaryFile()
            tf.write(str.encode(video_files))
            tf.flush()
            video_files = tf.name

            device_id = ParallelEnv().local_rank
            logger.info(f'---------- device_id: {device_id} -----------')

            pipe = VideoPipe(batch_size=self.batch_size,
                             num_threads=1,
                             device_id=device_id,
                             file_list=video_files,
                             sequence_length=self.num_seg * self.seglen,
                             num_seg=self.num_seg,
                             seg_length=self.seglen,
                             resize_shorter_scale=self.short_size,
                             crop_target_size=self.target_size,
                             is_training=True,
                             num_shards=self.num_shards,
                             shard_id=self.shard_id,
                             dali_mean=self.dali_mean,
                             dali_std=self.dali_std)

            logger.info(
                'initializing dataset, it will take several minutes if it is too large .... '
            )
            video_loader = DALIGenericIterator([pipe], ['image', 'label'],
                                               len(lines),
                                               dynamic_shape=True,
                                               auto_reset=True)

            return video_loader

        dali_reader = reader_()
        return dali_reader


class VideoPipe(Pipeline):
    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 file_list,
                 sequence_length,
                 num_seg,
                 seg_length,
                 resize_shorter_scale,
                 crop_target_size,
                 is_training=False,
                 initial_prefetch_size=20,
                 num_shards=1,
                 shard_id=0,
                 dali_mean=0.,
                 dali_std=1.0):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.VideoReader(device="gpu",
                                     file_list=file_list,
                                     sequence_length=sequence_length,
                                     num_seg=num_seg,
                                     seg_length=seg_length,
                                     is_training=is_training,
                                     num_shards=num_shards,
                                     shard_id=shard_id,
                                     random_shuffle=is_training,
                                     initial_fill=initial_prefetch_size)
        # the sequece data read by ops.VideoReader is of shape [F, H, W, C]
        # Because the ops.Resize does not support sequence data,
        # it will be transposed into [H, W, F, C],
        # then reshaped to [H, W, FC], and then resized like a 2-D image.
        self.transpose = ops.Transpose(device="gpu", perm=[1, 2, 0, 3])
        self.reshape = ops.Reshape(device="gpu",
                                   rel_shape=[1.0, 1.0, -1],
                                   layout='HWC')
        self.resize = ops.Resize(device="gpu",
                                 resize_shorter=resize_shorter_scale)
        # crops and mirror are applied by ops.CropMirrorNormalize.
        # Normalization will be implemented in paddle due to the difficulty of dimension broadcast,
        # It is not sure whether dimension broadcast can be implemented correctly by dali, just take the Paddle Op instead.
        self.pos_rng_x = ops.Uniform(range=(0.0, 1.0))
        self.pos_rng_y = ops.Uniform(range=(0.0, 1.0))
        self.mirror_generator = ops.Uniform(range=(0.0, 1.0))
        self.cast_mirror = ops.Cast(dtype=types.DALIDataType.INT32)
        self.crop_mirror_norm = ops.CropMirrorNormalize(
            device="gpu",
            crop=[crop_target_size, crop_target_size],
            mean=dali_mean,
            std=dali_std)
        self.reshape_back = ops.Reshape(
            device="gpu",
            shape=[num_seg, seg_length * 3, crop_target_size, crop_target_size],
            layout='FCHW')
        self.cast_label = ops.Cast(device="gpu", dtype=types.DALIDataType.INT64)

    def define_graph(self):
        output, label = self.input(name="Reader")
        output = self.transpose(output)
        output = self.reshape(output)

        output = self.resize(output)
        output = output / 255.
        pos_x = self.pos_rng_x()
        pos_y = self.pos_rng_y()
        mirror_flag = self.mirror_generator()
        mirror_flag = (mirror_flag > 0.5)
        mirror_flag = self.cast_mirror(mirror_flag)
        output = self.crop_mirror_norm(output,
                                       crop_pos_x=pos_x,
                                       crop_pos_y=pos_y,
                                       mirror=mirror_flag)
        output = self.reshape_back(output)
        label = self.cast_label(label)
        return output, label

    def __len__(self):
        return self.epoch_size()
