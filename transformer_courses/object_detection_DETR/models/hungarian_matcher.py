from scipy.optimize import linear_sum_assignment
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.util import bbox_cxcywh_to_xyxy
from utils.util import GIoULoss

class HungarianMatcher(nn.Layer):
    def __init__(self,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self, boxes, logits, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = paddle.gather(
                pos_cost_class, tgt_ids, axis=1) - paddle.gather(
                    neg_cost_class, tgt_ids, axis=1)
        else:
            cost_class = -paddle.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]

        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c.split(sizes, -1)[i].numpy())
            for i, c in enumerate(C)
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]
