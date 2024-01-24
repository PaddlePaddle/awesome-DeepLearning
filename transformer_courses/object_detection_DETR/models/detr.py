import paddle
import paddle.nn as nn

class DETR(nn.Layer):
    def __init__(self,
                 backbone,
                 transformer,
                 detr_head,
                 post_process='DETRBBoxPostProcess',
                 data_format='NCHW'):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.post_process = post_process
        self.data_format = data_format

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])
        self.inputs = inputs
        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            out = self.get_pred()
        return out

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        out_transformer = self.transformer(body_feats, self.inputs['pad_mask'])

        # DETR Head
        if self.training:
            return self.detr_head(out_transformer, body_feats, self.inputs)
        else:
            preds = self.detr_head(out_transformer, body_feats)
            bbox, bbox_num = self.post_process(preds, self.inputs['im_shape'],
                                               self.inputs['scale_factor'])
            return bbox, bbox_num

    def build_inputs(self, data, input_def):
        inputs = {}
        for i, k in enumerate(input_def):
            inputs[k] = data[i]
        return inputs
    
    def model_arch(self, ):
        pass

    def get_loss(self, ):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {
            "bbox": bbox_pred,
            "bbox_num": bbox_num,
        }
        return output