import numpy as np
import time
import collections
import paddle
import paddle.distributed as dist

from data.operators import *
from data import COCODataSet, BaseDataLoader
from models import PiecewiseDecay, LearningRate, OptimizerBuilder
from models import ComposeCallback, LogPrinter, Checkpointer


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({avg:.4f})"
        self.deque = collections.deque(maxlen=window_size)
        self.fmt = fmt
        self.total = 0.
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return np.mean(self.deque)

    @property
    def max(self):
        return np.max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, max=self.max, value=self.value)


class TrainingStats(object):
    def __init__(self, window_size, delimiter=' '):
        self.meters = None
        self.window_size = window_size
        self.delimiter = delimiter

    def update(self, stats):
        if self.meters is None:
            self.meters = {
                k: SmoothedValue(self.window_size)
                for k in stats.keys()
            }
        for k, v in self.meters.items():
            v.update(stats[k].numpy())

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.meters.items():
            stats[k] = format(v.median, '.6f')

        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = []
        for k, v in d.items():
            strs.append("{}: {}".format(k, str(v)))
        return self.delimiter.join(strs)

def train(model, start_epoch, epoch,dataset_dir,image_dir,anno_path):
    status = {}
    batch_size = 16
    _nranks = dist.get_world_size()
    _local_rank = dist.get_rank()

    # 读取训练集
    dataset = COCODataSet(dataset_dir=dataset_dir, image_dir=image_dir,anno_path=anno_path,data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'])
    sample_transforms = [{Decode: {}}, {RandomFlip: {'prob': 0.5}}, {RandomSelect: {'transforms1': [{RandomShortSideResize: {'short_side_sizes': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], 'max_size': 1333}}], 'transforms2': [{RandomShortSideResize: {'short_side_sizes': [400, 500, 600]}}, {RandomSizeCrop: {'min_size': 384, 'max_size': 600}}, {RandomShortSideResize: {'short_side_sizes': [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], 'max_size': 1333}}]}}, {NormalizeImage: {'is_scale': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, {NormalizeBox: {}}, {BboxXYXY2XYWH: {}}, {Permute: {}}]
    batch_transforms = [{PadMaskBatch: {'pad_to_stride': -1, 'return_pad_mask': True}}]    
    loader = BaseDataLoader(sample_transforms, batch_transforms, batch_size=2, shuffle=True, drop_last=True,collate_batch=False, use_shared_memory=False)(
        dataset, 0)
    # build optimizer in train mode
    steps_per_epoch = len(loader)


    # 设置学习率、优化器
    schedulers = PiecewiseDecay(gamma=0.1,milestones=[400],use_warmup=False)
    lr_ = LearningRate(base_lr=0.0001, schedulers=schedulers)
    optimizer_ = OptimizerBuilder(clip_grad_by_norm=0.1, regularizer=False, optimizers={'type': 'AdamW', 'weight_decay': 0.0001})
    lr = lr_(steps_per_epoch)
    optimizers = optimizer_(lr,model.parameters())

    # initial default callbacks
    _callbacks = [LogPrinter(model,batch_size), Checkpointer(model,optimizers)]
    _compose_callback = ComposeCallback(_callbacks)


    if _nranks > 1:
        model = paddle.DataParallel(model, find_unused_parameters=False)


    status.update({
        'epoch_id': start_epoch,
        'step_id': 0,
        'steps_per_epoch': len(loader)
    })
    
    status['batch_time'] = SmoothedValue(20, fmt='{avg:.4f}')
    status['data_time'] = SmoothedValue(20, fmt='{avg:.4f}')
    status['training_staus'] = TrainingStats(20)


    for epoch_id in range(start_epoch, epoch):
        status['mode'] = 'train'
        status['epoch_id'] = epoch_id
        _compose_callback.on_epoch_begin(status)
        loader.dataset.set_epoch(epoch_id)
        model.train()
        iter_tic = time.time()
        for step_id, data in enumerate(loader):
            status['data_time'].update(time.time() - iter_tic)
            status['step_id'] = step_id
            _compose_callback.on_step_begin(status)

            # model forward
            outputs = model(data)
            loss = outputs['loss']
            # model backward
            loss.backward()
            optimizers.step()

            curr_lr = optimizers.get_lr()
            lr.step()
            optimizers.clear_grad()
            status['learning_rate'] = curr_lr

            if _nranks < 2 or _local_rank == 0:
                status['training_staus'].update(outputs)

            status['batch_time'].update(time.time() - iter_tic)
            _compose_callback.on_step_end(status)
            iter_tic = time.time()

        _compose_callback.on_epoch_end(status)