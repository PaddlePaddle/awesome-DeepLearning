from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

from paddle.fluid import layers

import utils
import contextlib
import codecs
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.profiler as profiler
import reader
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from args import parse_args
from language_model import lm_model
from config import RNNConfig
from paddle.fluid.executor import Executor
logger = utils.lm_logger
SEED = 123


def build_train_model(main_program, startup_program, config):
    """
    使用异步读取方式读取数据，然后构建网络
    :param main_program:
    :param startup_program:
    :param config:
    :return:
    """
    with fluid.program_guard(main_program, startup_program):
        feed_shapes = [[config.batch_size, config.num_steps, 1],
                       [config.batch_size * config.num_steps, 1]]
        py_reader = fluid.layers.py_reader(
            capacity=16, shapes=feed_shapes, dtypes=['int64', 'int64'])
        x, y = fluid.layers.read_file(py_reader)
        # 使用 unique_name.guard 创建变量“生存空间”，以便和 infer 共享参数
        with fluid.unique_name.guard():
            res_vars = lm_model(
                config.hidden_size,
                config.vocab_size,
                config.batch_size,
                num_layers=config.num_layers,
                num_steps=config.num_steps,
                init_scale=config.init_scale,
                dropout=config.dropout,
                rnn_model=config.rnn_model,
                x=x)

            projection, last_hidden, last_cell = res_vars
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=config.max_grad_norm))

            learning_rate = fluid.layers.create_global_var(
                name="learning_rate",
                shape=[1],
                value=1.0,
                dtype='float32',
                persistable=True)

            loss = get_loss(projection, y, config.num_steps)
            optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
            optimizer.minimize(loss)

            # 目前 paddle 设置了内存优化，如果没有设置可读写存储，
            # 会出现大量 warning 日志...
            loss.persistable = True
            projection.persistable = True
            last_hidden.persistable = True
            last_cell.persistable = True
            return loss, projection, last_hidden, last_cell, py_reader


def build_infer_model(main_program, startup_program, config):
    """
    构建预测验证的模型
    :param main_program:
    :param startup_program:
    :param config:
    :return:
    """
    with fluid.program_guard(main_program, startup_program):
        x = layers.data(
            name="x",
            shape=[config.batch_size, config.num_steps, 1],
            dtype='int64',
            append_batch_size=False)
        y = layers.data(
            name="y",
            shape=[config.batch_size * config.num_steps, 1],
            dtype='int64',
            append_batch_size=False)
        # 使用 unique_name.guard 创建变量“生存空间”，以便和 train 共享参数
        with fluid.unique_name.guard():
            res_vars = lm_model(
                config.hidden_size,
                config.vocab_size,
                config.batch_size,
                num_layers=config.num_layers,
                num_steps=config.num_steps,
                init_scale=config.init_scale,
                dropout=config.dropout,
                rnn_model=config.rnn_model,
                x=x)
            projection, last_hidden, last_cell = res_vars
            loss = get_loss(projection, y, config.num_steps)

            # 目前 paddle 设置了内存优化，如果没有设置可读写存储，
            # 会出现大量 warning 日志...
            loss.persistable = True
            projection.persistable = True
            last_hidden.persistable = True
            last_cell.persistable = True
            return loss, projection, last_hidden, last_cell


def get_loss(projection, y, num_steps):
    loss = layers.softmax_with_cross_entropy(logits=projection, label=y, soft_label=False)
    loss = layers.reshape(loss, shape=[-1, num_steps], inplace=True)
    loss = layers.reduce_mean(loss, dim=[0])
    loss = layers.reduce_sum(loss)
    return loss


def get_log_interval(data_len, config):
    num_batchs = data_len // config.batch_size
    epoch_size = (num_batchs - 1) // config.num_steps
    log_interval = max(1, epoch_size // 10)
    return log_interval


def generate_init_data(config):
    init_hidden = np.zeros(
        (config.num_layers, config.batch_size, config.hidden_size),
        dtype='float32')
    init_cell = np.zeros(
        (config.num_layers, config.batch_size, config.hidden_size),
        dtype='float32')
    return init_hidden, init_cell


def generate_new_lr(config, epoch_id=0, device_count=1):
    new_lr = config.base_learning_rate * (config.lr_decay**max(
        epoch_id + 1 - config.epoch_start_decay, 0.0))
    lr = np.ones((device_count), dtype='float32') * new_lr
    return lr


def prepare_input(config, batch, init_hidden=None,
                  init_cell=None, epoch_id=0,
                  with_lr=True, device_count=1):
    x, y = batch
    # 对于 RNN 来说，通常是一个batch拉通训练，而每一个timestamp输入都有一个输出，所以需要把y拉通
    x = x.reshape((-1, config.num_steps, 1))
    y = y.reshape((-1, 1))

    res = {'x': x, 'y': y}
    if init_hidden is not None:
        res['init_hidden'] = init_hidden
    if init_cell is not None:
        res['init_cell'] = init_cell
    if with_lr:
        res['learning_rate'] = generate_new_lr(config, epoch_id, device_count)

    return res


def train_an_epoch_py_reader(config, train_data, epoch_id, batch_times, py_reader,
                             train_program, loss, last_hidden, last_cell, exe):
    # get train epoch size
    log_interval = get_log_interval(len(train_data), config)

    init_hidden, init_cell = generate_init_data(config)

    total_loss = 0
    iters = 0

    py_reader.start()
    batch_id = 0
    batch_start_time = time.time()
    try:
        while True:
            data_feeds = {}
            new_lr = generate_new_lr(config, epoch_id)
            data_feeds['learning_rate'] = new_lr
            data_feeds["init_hidden"] = init_hidden
            data_feeds["init_cell"] = init_cell

            fetch_outs = exe.run(train_program,
                                 feed=data_feeds,
                                 fetch_list=[
                                     loss.name, "learning_rate",
                                     last_hidden.name, last_cell.name
                                 ],
                                 use_program_cache=True)

            cost_train = np.array(fetch_outs[0])
            lr = np.array(fetch_outs[1])
            init_hidden = np.array(fetch_outs[2])
            init_cell = np.array(fetch_outs[3])

            total_loss += cost_train
            iters += config.num_steps
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            batch_start_time = time.time()
            if batch_id > 0 and (log_interval == 0 or
                                 batch_id % log_interval == 0):
                ppl = np.exp(total_loss / iters)
                logger.info("-- Epoch:[%d]; Batch:[%d]; Time: %.5f s; ppl: %.5f, lr: %.5f"
                    % (epoch_id, batch_id, batch_time, ppl[0], lr[0]))

            batch_id += 1
    except fluid.core.EOFException:
        py_reader.reset()

    batch_times.append(time.time() - batch_start_time)
    ppl = np.exp(total_loss / iters)
    return ppl


def eval_date_feeder(data, config, inference_program, loss, last_hidden, last_cell, exe):
    # when eval the batch_size set to 1
    eval_data_iter = reader.get_data_iter(data, config.batch_size, config.num_steps)
    total_loss = 0.0
    iters = 0
    init_hidden, init_cell = generate_init_data(config)
    for batch_id, batch in enumerate(eval_data_iter):
        input_data_feed = prepare_input(config, batch, init_hidden, init_cell, epoch_id=0, with_lr=False)
        fetch_outs = exe.run(
            program=inference_program,
            feed=input_data_feed,
            fetch_list=[loss.name, last_hidden.name, last_cell.name],
            use_program_cache=False)

        cost_eval = np.array(fetch_outs[0])
        init_hidden = np.array(fetch_outs[1])
        init_cell = np.array(fetch_outs[2])

        total_loss += cost_eval
        iters += config.num_steps

    ppl = np.exp(total_loss / iters)
    return ppl


# NOTE(zjl): sometimes we have not enough data for eval if batch_size is large, i.e., 2100
# Just skip to avoid error
def is_valid_data(data, batch_size, num_steps):
    data_len = len(data)
    batch_len = data_len // batch_size
    epoch_size = (batch_len - 1) // num_steps
    return epoch_size >= 1


def main():
    args = parse_args()
    
    logger.info('Running with args : {}'.format(args))
    utils.check_cuda(args.use_gpu)

    config = RNNConfig(args)

    # 定义训练的 program
    main_program = fluid.Program()
    startup_program = fluid.Program()
    train_loss, train_proj, last_hidden, last_cell, py_reader = build_train_model(main_program, startup_program, config)

    # 定义预测的 program
    inference_program = fluid.Program()
    inference_startup_program = fluid.Program()
    infer_loss, infer_proj, _, _ = build_infer_model(inference_program, inference_startup_program, config)
    inference_program = inference_program.clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(startup_program)

    # 构建训练时候优化过的 program
    if args.parallel:
        device_count = len(fluid.cuda_places()) if args.use_gpu else len(fluid.cpu_places())

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = device_count
        exec_strategy.num_iteration_per_drop_scope = 100

        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = False
        build_strategy.fuse_all_optimizer_ops = True
        train_program = fluid.compiler.CompiledProgram(main_program).with_data_parallel(
                loss_name=train_loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
    else:
        train_program = fluid.compiler.CompiledProgram(main_program)

    data_path = args.data_path
    logger.info("begin to load data")
    vocab = reader.build_vocab(os.path.join(data_path, args.train_file))
    # 把词典保存下来
    with codecs.open('vocab.txt', 'w') as f:
        for (k, v) in vocab.items():
            f.write("{}\t{}\n".format(k, v))
    train_data = reader.load_data(data_path, args.train_file, vocab)
    valid_data = reader.load_data(data_path, args.eval_file, vocab)
    test_data = reader.load_data(data_path, args.test_file, vocab)
    logger.info("finished load data")

    # 对于 RNN 来说，通常是一个batch拉通训练，而每一个timestamp输入都有一个输出，
    # 所以需要把y拉通成一个数组，对每一个输出配对一个标签
    def data_gen():
        data_iter_size = config.batch_size // device_count
        train_batches = reader.get_data_iter(train_data, data_iter_size,
                                             config.num_steps)
        for batch in train_batches:
            x, y = batch
            x = x.reshape((-1, config.num_steps, 1))
            y = y.reshape((-1, 1))
            yield x, y

    py_reader.decorate_tensor_provider(data_gen)

    total_time = 0.0
    for epoch_id in range(config.max_epoch):
        batch_times = []
        epoch_start_time = time.time()
        train_ppl = train_an_epoch_py_reader(config, train_data, epoch_id, batch_times, py_reader,
                                             train_program, train_loss, last_hidden, last_cell, exe)
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        logger.info(
            "\nTrain epoch:[%d]; epoch Time: %.5f; ppl: %.5f; avg_time: %.5f steps/s \n"
            % (epoch_id, epoch_time, train_ppl[0],
               len(batch_times) / sum(batch_times)))

        # FIXME(zjl): ppl[0] increases as batch_size increases.
        # We should find a better way to calculate ppl by normalizing batch_size.
        if device_count == 1 and config.batch_size <= 20 and epoch_id == 0 and train_ppl[0] > 1000:
            # for bad init, after first epoch, the loss is over 1000
            # no more need to continue
            logger.info(
                "Parameters are randomly initialized and not good this time because the loss is over 1000 after the first epoch."
            )
            logger.info("Abort this training process and please start again.")
            return

        if epoch_id == config.max_epoch - 1 and args.enable_ce:
            # kpis
            logger.info("ptblm\tlstm_language_model_%s_duration_card%d\t%s" % (
            args.rnn_model, device_count, total_time / config.max_epoch))
            logger.info("ptblm\tlstm_language_model_%s_loss_card%d\t%s" % (args.rnn_model, device_count, train_ppl[0]))

        valid_data_valid = is_valid_data(valid_data, config.batch_size, config.num_steps)
        if valid_data_valid:
            valid_ppl = eval_date_feeder(valid_data, config, inference_program, infer_loss, last_hidden, last_cell, exe)
            logger.info("Valid ppl: %.5f" % valid_ppl[0])
        else:
            logger.info(
                'WARNING: length of valid_data is {}, which is not enough for batch_size {} and num_steps {}'.
                format(len(valid_data), config.batch_size, config.num_steps))

        save_model_dir = args.save_model_dir
        fluid.io.save_persistables(executor=exe, dirname=save_model_dir, main_program=main_program)
        logger.info("Saved model to: %s." % save_model_dir)

    test_ppl = eval_date_feeder(test_data, config, inference_program, infer_loss, last_hidden, last_cell, exe)
    logger.info("Test ppl:{}".format(test_ppl[0]))


if __name__ == '__main__':
    main()