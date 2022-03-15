from abc import abstractmethod
from ... import builder
import paddle.nn as nn
from ...registry import DETECTORS

@DETECTORS.register()
class BaseDetector(nn.Layer):
    """Base class for detectors.  """
    def __init__(self, backbone=None, head=None):

        super().__init__()

    def init_weights(self):
        """Initialize the model network weights. """
        self.backbone.init_weights()  
        self.head.init_weights()

    def extract_feature(self, imgs, iter_num):
        """Extract features through a backbone.  """
        feature = self.backbone(imgs)
        return feature

    def forward(self,  data_batch, mode='infer'):
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
