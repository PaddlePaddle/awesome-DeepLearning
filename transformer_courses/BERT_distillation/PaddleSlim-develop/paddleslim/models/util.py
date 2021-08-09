from __future__ import absolute_import
import paddle.fluid as fluid
from ..models import classification_models

__all__ = ["image_classification"]

model_list = classification_models.model_list


def image_classification(model, image_shape, class_num, use_gpu=False):
    assert model in model_list
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        model = classification_models.__dict__[model]()
        out = model.net(input=image, class_dim=class_num)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        val_program = fluid.default_main_program().clone(for_test=True)

        opt = fluid.optimizer.Momentum(0.1, 0.9)
        opt.minimize(avg_cost)
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
    return exe, train_program, val_program, (image, label), (
        acc_top1.name, acc_top5.name, avg_cost.name, out.name)
