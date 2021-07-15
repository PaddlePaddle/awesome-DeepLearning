import numpy as np
import logging
import paddle
from paddle.fluid.dygraph import TracedLayer
from paddleslim.core import GraphWrapper, dygraph2program
from paddleslim.prune import PruningCollections
from paddleslim.common import get_logger

__all__ = ["DygraphPruningCollections"]

_logger = get_logger(__name__, level=logging.INFO)


class DygraphPruningCollections(PruningCollections):
    """
    A tool used to parse dygraph and store information of variables' relationship.
    Args:
      - model(nn.Layer): The dygraph to be parsed.
      - inputs(Variable|list|dict): The dummy inputs of target model. It will be used in calling `model.forward(inputs)`.
    """

    def __init__(self, model, inputs):
        _logger.debug("Parsing model with input: {}".format(inputs))
        # model can be in training mode, because some model contains auxiliary parameters for training.
        program = dygraph2program(model, inputs=inputs)
        graph = GraphWrapper(program)
        params = [
            _param.name for _param in model.parameters()
            if len(_param.shape) == 4
        ]
        self._collections = self.create_pruning_collections(params, graph)
        _logger.info("Found {} collections.".format(len(self._collections)))

        _name2values = {}
        for param in model.parameters():
            _name2values[param.name] = np.array(param.value().get_tensor())
        for collection in self._collections:
            collection.values = _name2values

    def find_collection_by_master(self, var_name, axis):
        for _collection in self._collections:
            if _collection.master['name'] == var_name and _collection.master[
                    'axis'] == axis:
                return _collection

    def __str__(self):
        return "\n".join(
            [str(_collection) for _collection in self._collections])
