import paddle
import paddle.nn.functional as F

def bbox_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)

def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    prob = F.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss.mean(1).sum() / normalizer


# DETR调用
def inverse_sigmoid(x, eps=1e-6):
    x = x.clip(min=0., max=1.)
    return paddle.log(x / (1 - x + eps) + eps)

class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clip(0)
        h_inter = (ykis2 - ykis1).clip(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = paddle.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = paddle.sum(giou * iou_weight)
        else:
            loss = paddle.mean(giou * iou_weight)
        return loss * self.loss_weight