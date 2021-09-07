import paddle
import paddle.nn.functional as F

from utils.util import bbox_cxcywh_to_xyxy

class DETRBBoxPostProcess(object):
    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 use_focal_loss=False):
        super(DETRBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    def __call__(self, head_out, im_shape, scale_factor):
        """
        Decode the bbox.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = origin_shape.unbind(1)
        origin_shape = paddle.stack(
            [img_w, img_h, img_w, img_h], axis=-1).unsqueeze(0)
        bbox_pred *= origin_shape

        scores = F.sigmoid(logits) if self.use_focal_loss else F.softmax(
            logits)[:, :, :-1]

        if not self.use_focal_loss:
            scores, labels = scores.max(-1), scores.argmax(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = paddle.topk(
                    scores, self.num_top_queries, axis=-1)
                labels = paddle.stack(
                    [paddle.gather(l, i) for l, i in zip(labels, index)])
                bbox_pred = paddle.stack(
                    [paddle.gather(b, i) for b, i in zip(bbox_pred, index)])
        else:
            scores, index = paddle.topk(
                scores.reshape([logits.shape[0], -1]),
                self.num_top_queries,
                axis=-1)
            labels = index % logits.shape[2]
            index = index // logits.shape[2]
            bbox_pred = paddle.stack(
                [paddle.gather(b, i) for b, i in zip(bbox_pred, index)])

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype('float32'), scores.unsqueeze(-1),
                bbox_pred
            ],
            axis=-1)
        bbox_num = paddle.to_tensor(
            bbox_pred.shape[1], dtype='int32').tile([bbox_pred.shape[0]])
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num
