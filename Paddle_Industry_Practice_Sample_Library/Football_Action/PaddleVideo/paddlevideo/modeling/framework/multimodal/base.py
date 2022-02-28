from abc import abstractmethod
from ... import builder
import paddle.nn as nn


class BaseMultimodal(nn.Layer):
    """Base class for Multimodal.

    All Multimodal model should subclass it.
    All subclass should overwrite:

    - Methods:``train_step``, supporting to forward when training.
    - Methods:``valid_step``, supporting to forward when validating.
    - Methods:``test_step``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Head to process feature.
        loss(dict): Loss function.

    """
    def __init__(self, backbone=None, head=None, loss=None):
        super().__init__()
        if backbone is not None:
            self.backbone = builder.build_backbone(backbone)
            if hasattr(self.backbone, 'init_weights'):
                self.backbone.init_weights()
        else:
            self.backbone = None
        if head is not None:
            self.head_name = head.name
            self.head = builder.build_head(head)
            if hasattr(self.head, 'init_weights'):
                self.head.init_weights()
        else:
            self.head = None
        if loss is not None:
            self.loss = builder.build_loss(loss)
        else:
            self.loss = None

    def forward(self, data_batch, mode='infer'):
        """
        1. Define how the model is going to run, from input to output.
        2. Console of train, valid, test or infer step
        3. Set mode='infer' is used for saving inference model, refer to tools/export_model.py
        """
        if mode == 'train':
            return self.train_step(data_batch)
        elif mode == 'valid':
            return self.val_step(data_batch)
        elif mode == 'test':
            return self.test_step(data_batch)
        elif mode == 'infer':
            return self.infer_step(data_batch)
        else:
            raise NotImplementedError

    @abstractmethod
    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data_batch, **kwargs):
        """Validating step.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data_batch, **kwargs):
        """Test step.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_step(self, data_batch, **kwargs):
        """Infer step.
        """
        raise NotImplementedError
