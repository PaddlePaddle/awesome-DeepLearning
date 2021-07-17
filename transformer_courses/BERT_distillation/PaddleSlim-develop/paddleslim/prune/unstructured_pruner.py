import numpy as np
from ..common import get_logger
from ..core import GraphWrapper
import paddle

__all__ = ["UnstructuredPruner"]


class UnstructuredPruner():
    """
    The unstructure pruner.

    Args:
      - program(paddle.static.Program): The model to be pruned.
      - mode(str): the mode to prune the model, must be selected from 'ratio' and 'threshold'.
      - ratio(float): the ratio to prune the model. Only set it when mode=='ratio'. Default: 0.5.
      - threshold(float): the threshold to prune the model. Only set it when mode=='threshold'. Default: 1e-5.
      - scope(paddle.static.Scope): The scope storing values of all variables. None means paddle.static.global_scope. Default: None.
      - place(CPUPlace | CUDAPlace): The device place used to execute model. None means CPUPlace. Default: None.
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params.
    """

    def __init__(self,
                 program,
                 mode,
                 ratio=0.5,
                 threshold=1e-5,
                 scope=None,
                 place=None,
                 skip_params_func=None):
        self.mode = mode
        self.ratio = ratio
        self.threshold = threshold
        assert self.mode in [
            'ratio', 'threshold'
        ], "mode must be selected from 'ratio' and 'threshold'"
        self.scope = paddle.static.global_scope() if scope == None else scope
        self.place = paddle.static.CPUPlace() if place is None else place
        if skip_params_func is None: skip_params_func = self._get_skip_params
        self.skip_params = skip_params_func(program)
        self.masks = self._apply_masks(program)

    def _apply_masks(self, program):
        params = []
        masks = []
        for param in program.all_parameters():
            mask = program.global_block().create_var(
                name=param.name + "_mask",
                shape=param.shape,
                dtype=param.dtype,
                type=param.type,
                persistable=param.persistable,
                stop_gradient=True)

            self.scope.var(param.name + "_mask").get_tensor().set(
                np.ones(mask.shape).astype("float32"), self.place)
            params.append(param)
            masks.append(mask)

        d_masks = {}
        for _param, _mask in zip(params, masks):
            d_masks[_param.name] = _mask.name
        return d_masks

    def summarize_weights(self, program, ratio=0.1):
        """
        The function is used to get the weights corresponding to a given ratio
        when you are uncertain about the threshold in __init__() function above.
        For example, when given 0.1 as ratio, the function will print the weight value,
        the abs(weights) lower than which count for 10% of the total numbers.

        Args:
          - program(paddle.static.Program): The model which have all the parameters.
          - ratio(float): The ratio illustrated above.
        Return:
          - threshold(float): a threshold corresponding to the input ratio.
        """
        data = []
        for param in program.all_parameters():
            data.append(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()).flatten())
        data = np.concatenate(data, axis=0)
        threshold = np.sort(np.abs(data))[max(0, int(ratio * len(data) - 1))]
        return threshold

    def sparse_by_layer(self, program):
        """
        The function is used to get the density at each layer, usually called for debuggings.
        
        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - layer_sparse(Dict<string, float>): sparsity for each parameter.
        """
        layer_sparse = {}
        total = 0
        values = 0
        for param in program.all_parameters():
            value = np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
            layer_sparse[param.name] = value / np.product(param.shape)
        return layer_sparse

    def update_threshold(self):
        '''
        Update the threshold after each optimization step in RATIO mode.
        User should overwrite this method to define their own weight importance (Default is based on their absolute values).
        '''
        params_flatten = []
        for param in self.masks:
            if not self._should_prune_param(param):
                continue
            t_param = self.scope.find_var(param).get_tensor()
            v_param = np.array(t_param)
            params_flatten.append(v_param.flatten())
        params_flatten = np.concatenate(params_flatten, axis=0)
        total_len = len(params_flatten)
        self.threshold = np.sort(np.abs(params_flatten))[max(
            0, int(self.ratio * total_len) - 1)]

    def _update_params_masks(self):
        for param in self.masks:
            if not self._should_prune_param(param):
                continue
            mask_name = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask_name).get_tensor()
            v_param = np.array(t_param)
            v_param[np.abs(v_param) < self.threshold] = 0
            v_mask = (v_param != 0).astype(v_param.dtype)
            t_mask.set(v_mask, self.place)
            v_param = np.array(t_param) * np.array(t_mask)
            t_param.set(v_param, self.place)

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'threshold':
            pass
        elif self.mode == 'ratio':
            self.update_threshold()
        self._update_params_masks()

    def update_params(self):
        """
        Update the parameters given self.masks, usually called before saving models.
        """
        for param in self.masks:
            mask = self.masks[param]
            t_param = self.scope.find_var(param).get_tensor()
            t_mask = self.scope.find_var(mask).get_tensor()
            v_param = np.array(t_param) * np.array(t_mask)
            t_param.set(v_param, self.place)

    @staticmethod
    def total_sparse(program):
        """
        The function is used to get the whole model's density (1-sparsity).
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.

        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - density(float): the model's density.
        """
        total = 0
        values = 0
        for param in program.all_parameters():
            total += np.product(param.shape)
            values += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
        density = float(values) / total
        return density

    def _get_skip_params(self, program):
        """
        The function is used to get a set of all the skipped parameters when performing pruning.
        By default, the normalization-related ones will not be pruned.
        Developers could replace it by passing their own function when initializing the UnstructuredPruner instance.

        Args:
          - program(paddle.static.Program): the current model.
        Returns:
          - skip_params(Set<String>): a set of parameters' names.
        """
        skip_params = set()
        graph = GraphWrapper(program)
        for op in graph.ops():
            if 'norm' in op.type() and 'grad' not in op.type():
                for input in op.all_inputs():
                    skip_params.add(input.name())
        return skip_params

    def _should_prune_param(self, param):
        should_prune = param not in self.skip_params
        return should_prune
