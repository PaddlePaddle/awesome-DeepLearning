import numpy as np
import paddle
from paddlevideo.utils import get_logger

from .base import BaseMetric
from .registry import METRIC

logger = get_logger("paddlevideo")


@METRIC.register
class DepthMetric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.abs_rel = []
        self.sq_rel = []
        self.rmse = []
        self.rmse_log = []
        self.a1 = []
        self.a2 = []
        self.a3 = []

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = outputs['abs_rel'], outputs['sq_rel'], outputs['rmse'], \
                                                      outputs['rmse_log'], outputs['a1'], outputs['a2'],outputs['a3']
        # preds ensemble
        if self.world_size > 1:
            abs_rel = paddle.distributed.all_reduce(
                outputs['abs_rel'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            sq_rel = paddle.distributed.all_reduce(
                outputs['sq_rel'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            rmse = paddle.distributed.all_reduce(
                outputs['rmse'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            rmse_log = paddle.distributed.all_reduce(
                outputs['rmse_log'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            a1 = paddle.distributed.all_reduce(
                outputs['a1'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            a2 = paddle.distributed.all_reduce(
                outputs['a2'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            a3 = paddle.distributed.all_reduce(
                outputs['a3'],
                op=paddle.distributed.ReduceOp.SUM) / self.world_size

        self.abs_rel.append(abs_rel)
        self.sq_rel.append(sq_rel)
        self.rmse.append(rmse)
        self.rmse_log.append(rmse_log)
        self.a1.append(a1)
        self.a2.append(a2)
        self.a3.append(a3)
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        logger.info(
            '[TEST] finished, abs_rel= {}, sq_rel= {} , rmse= {}, rmse_log= {},'
            'a1= {}, a2= {}, a3= {}'.format(np.mean(np.array(self.abs_rel)),
                                            np.mean(np.array(self.sq_rel)),
                                            np.mean(np.array(self.rmse)),
                                            np.mean(np.array(self.rmse_log)),
                                            np.mean(np.array(self.a1)),
                                            np.mean(np.array(self.a2)),
                                            np.mean(np.array(self.a3))))
