import os
import paddle
from ..core import GraphWrapper
from ..common import get_logger
import json
import logging

__all__ = ["save_model", "load_model"]

_logger = get_logger(__name__, level=logging.INFO)

_SHAPES_FILE = "__shapes__"
_GROUPS_FILE = "__groups__"


def save_model(exe, graph, dirname):
    """
    Save weights of model and information of shapes into filesystem.

    Args:
        exe(paddle.static.Executor): The executor used to save model.
        graph(Program|Graph): The graph to be saved.
        dirname(str): The directory that the model saved into.
    """
    assert graph is not None and dirname is not None
    graph = GraphWrapper(graph) if isinstance(graph,
                                              paddle.static.Program) else graph

    paddle.static.save(program=graph.program, model_path=dirname)
    weights_file = dirname
    _logger.info("Save model weights into {}".format(weights_file))
    shapes = {}
    for var in graph.program.list_vars():
        if var.persistable and str(var.type) != 'VarType.READER':
            shapes[var.name] = var.shape
    SHAPES_FILE = os.path.join(dirname, _SHAPES_FILE)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(SHAPES_FILE, "w") as f:
        json.dump(shapes, f)
        _logger.info("Save shapes of weights into {}".format(SHAPES_FILE))

    groups = {}
    for op in graph.ops():
        if 'conv2d' in op.type():
            filter_name = op.inputs('Filter')[0].name()
            groups[filter_name] = op.attr('groups')

    GROUPS_FILE = os.path.join(dirname, _GROUPS_FILE)
    with open(GROUPS_FILE, "w") as f:
        json.dump(groups, f)
        _logger.info("Save groups of cnov2d into {}".format(GROUPS_FILE))


def load_model(exe, graph, dirname):
    """
    Load weights of model and information of shapes from filesystem.

    Args:
        graph(Program|Graph): The graph to be updated by loaded information..
        dirname(str): The directory that the model will be loaded.
    """
    assert graph is not None and dirname is not None
    graph = GraphWrapper(graph) if isinstance(graph,
                                              paddle.static.Program) else graph

    SHAPES_FILE = os.path.join(dirname, _SHAPES_FILE)
    with open(SHAPES_FILE, "r") as f:
        shapes = json.load(f)
        for param_name, shape in shapes.items():
            param = graph.var(param_name)
            if param is not None:
                param.set_shape(shape)
            else:
                _logger.info('{} is not loaded'.format(param_name))
    _logger.info("Load shapes of weights from {}".format(SHAPES_FILE))

    GROUPS_FILE = os.path.join(dirname, _GROUPS_FILE)
    with open(GROUPS_FILE, "r") as f:
        groups = json.load(f)
        for op in graph.ops():
            if 'conv2d' in op.type():
                filter_name = op.inputs('Filter')[0].name()
                op.set_attr('groups', groups[filter_name])
    _logger.info("Load groups of conv2d from {}".format(GROUPS_FILE))

    paddle.static.load(program=graph.program, model_path=dirname, executor=exe)
    graph.infer_shape()
    _logger.info("Load weights from {}".format(dirname))
