import numpy as np
import paddle
import logging
from paddleslim.common import get_logger

__all__ = ["UnstructuredPruner"]

_logger = get_logger(__name__, level=logging.INFO)

NORMS_ALL = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'BatchNorm1D',
    'BatchNorm2D', 'BatchNorm3D', 'InstanceNorm1D', 'InstanceNorm2D',
    'InstanceNorm3D', 'SyncBatchNorm', 'LocalResponseNorm'
]


class UnstructuredPruner():
    """
    The unstructure pruner.
    Args:
      - model(Paddle.nn.Layer): The model to be pruned.
      - mode(str): Pruning mode, must be selected from 'ratio' and 'threshold'.
      - threshold(float): The parameters whose absolute values are smaller than the THRESHOLD will be zeros. Default: 0.01
      - ratio(float): The parameters whose absolute values are in the smaller part decided by the ratio will be zeros. Default: 0.3
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params. 
    """

    def __init__(self,
                 model,
                 mode,
                 threshold=0.01,
                 ratio=0.3,
                 skip_params_func=None):
        assert mode in ('ratio', 'threshold'
                        ), "mode must be selected from 'ratio' and 'threshold'"
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.ratio = ratio
        if skip_params_func is None: skip_params_func = self._get_skip_params
        self.skip_params = skip_params_func(model)
        self._apply_masks()

    def mask_parameters(self, param, mask):
        """
        Update masks and parameters. It is executed to each layer before each iteration.
        User can overwrite this function in subclass to implememt different pruning stragies.
        Args:
          - parameters(list<Tensor>): The parameters to be pruned.
          - masks(list<Tensor>): The masks used to keep zero values in parameters.
        """
        bool_tmp = (paddle.abs(param) >= self.threshold)
        paddle.assign(mask * bool_tmp, output=mask)
        param_tmp = param * mask
        param_tmp.stop_gradient = True
        paddle.assign(param_tmp, output=param)

    def _apply_masks(self):
        self.masks = {}
        for name, sub_layer in self.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                tmp_array = np.ones(param.shape, dtype=np.float32)
                mask_name = "_".join([param.name.replace(".", "_"), "mask"])
                if mask_name not in sub_layer._buffers:
                    sub_layer.register_buffer(mask_name,
                                              paddle.to_tensor(tmp_array))
                self.masks[param.name] = sub_layer._buffers[mask_name]
        for name, sub_layer in self.model.named_sublayers():
            sub_layer.register_forward_pre_hook(self._forward_pre_hook)

    def update_threshold(self):
        '''
        Update the threshold after each optimization step.
        User should overwrite this method togther with self.mask_parameters()
        '''
        params_flatten = []
        for name, sub_layer in self.model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                t_param = param.value().get_tensor()
                v_param = np.array(t_param)
                params_flatten.append(v_param.flatten())
        params_flatten = np.concatenate(params_flatten, axis=0)
        total_length = params_flatten.size
        self.threshold = np.sort(np.abs(params_flatten))[max(
            0, round(self.ratio * total_length) - 1)].item()

    def summarize_weights(self, model, ratio=0.1):
        """
        The function is used to get the weights corresponding to a given ratio
        when you are uncertain about the threshold in __init__() function above.
        For example, when given 0.1 as ratio, the function will print the weight value,
        the abs(weights) lower than which count for 10% of the total numbers.
        Args:
          - model(paddle.nn.Layer): The model which have all the parameters.
          - ratio(float): The ratio illustrated above.
        Return:
          - threshold(float): a threshold corresponding to the input ratio.
        """
        data = []
        for name, sub_layer in model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                data.append(np.array(param.value().get_tensor()).flatten())
        data = np.concatenate(data, axis=0)
        threshold = np.sort(np.abs(data))[max(0, int(ratio * len(data) - 1))]
        return threshold

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'ratio':
            self.update_threshold()
        elif self.mode == 'threshold':
            return

    def _forward_pre_hook(self, layer, input):
        if not self._should_prune_layer(layer):
            return input
        for param in layer.parameters(include_sublayers=False):
            mask = self.masks.get(param.name)
            self.mask_parameters(param, mask)
        return input

    def update_params(self):
        """
        Update the parameters given self.masks, usually called before saving models and evaluation step during training. 
        If you load a sparse model and only want to inference, no need to call the method.
        """
        for name, sub_layer in self.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                mask = self.masks.get(param.name)
                param_tmp = param * mask
                param_tmp.stop_gradient = True
                paddle.assign(param_tmp, output=param)

    @staticmethod
    def total_sparse(model):
        """
        This static function is used to get the whole model's density (1-sparsity).
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        
        Args:
          - model(paddle.nn.Layer): The sparse model.
        Returns:
          - ratio(float): The model's density.
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = float(values) / total
        return ratio

    def _get_skip_params(self, model):
        """
        This function is used to check whether the given model's layers are valid to be pruned. 
        Usually, the convolutions are to be pruned while we skip the normalization-related parameters.
        Deverlopers could replace this function by passing their own when initializing the UnstructuredPuner instance.

        Args:
          - model(Paddle.nn.Layer): the current model waiting to be checked.
        Return:
          - skip_params(set<String>): a set of parameters' names
        """
        skip_params = set()
        for _, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                skip_params.add(sub_layer.full_name())
        return skip_params

    def _should_prune_layer(self, layer):
        should_prune = layer.full_name() not in self.skip_params
        return should_prune
