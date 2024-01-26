

import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUF_SIZE = 500
BATCH_SIZE = 20

# paddle.reader.shuffle()表示每次缓存BUF_SIZE个数据项，并进行打乱
# paddle.batch()表示每BATCH_SIZE组成一个batch
# 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
# 用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)

# 用于打印，查看uci_housing数据
train_data = paddle.dataset.uci_housing.train();
sampledata = next(train_data())
print(sampledata)

# 定义张量变量x，表示13维的特征值
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
# 定义张量y,表示目标值
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
# 定义一个简单的线性网络,连接输入和输出的全连接层
# input:输入tensor;
# size:该层输出单元的数目
# act:激活函数
y_predict = fluid.layers.fc(input=x, size=1, act=None)

# 定义损失函数
# 求一个batch的损失值，input=y_predict表示预测值，label=y表示实际值
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)  # 对损失值求平均值

# 定义优化函数    使用随机梯度下降法SGD
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

test_program = fluid.default_main_program().clone(for_test=True)

# use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()  # 使用CPU运算
# 创建一个Executor实例exe
exe = fluid.Executor(place)
# Executor的run()方法执行startup_program(),进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度   feed_list:向模型输入的变量表或变量表名
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

iter = 0;
iters = []
train_costs = []


def draw_train_process(iters, train_costs):
    title = "training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.grid()
    plt.show()


# 训练并保存模型
EPOCH_NUM = 50
model_save_dir = "C:/Users/nxy/Desktop/fit_a_line.inference.model"

for pass_id in range(EPOCH_NUM):  # 训练EPOCH_NUM轮
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader迭代器
        train_cost = exe.run(program=fluid.default_main_program(),  # 运行主程序
                             # 喂入一个batch的训练数据，根据feed_list和data提供的信息，将输入数据转成一种特殊的数据结构
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
        if batch_id % 40 == 0:
            # 打印最后一个batch的损失值
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))
        iter = iter + BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])

    # 开始测试并输出最后一个batch的损失值
    test_cost = 0
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader迭代器
        test_cost = exe.run(program=test_program,  # 运行测试cheng
                            feed=feeder.feed(data),  # 喂入一个batch的测试数据
                            fetch_list=[avg_cost])  # fetch均方误差
    # 打印最后一个batch的损失值
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))

    # 保存模型
    # 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
# 保存训练参数到指定路径中，构建一个专门用预测的program
fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                              ['x'],  # 推理（inference）需要 feed 的数据
                              [y_predict],  # 保存推理（inference）结果的 Variables
                              exe)  # exe 保存 inference model
draw_train_process(iters, train_costs)

# 模型预测
infer_exe = fluid.Executor(place)  # 创建推测用的executor
inference_scope = fluid.core.Scope()  # Scope指定作用域

# 可视化真实值与预测值方法定义
infer_results = []
groud_truths = []


# 绘制真实值和预测值对比图
def draw_infer_result(groud_truths, infer_results):
    title = 'Boston'
    plt.title(title, fontsize=24)
    x = np.arange(1, 20)
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results, color='green', label='training cost')
    plt.grid()
    plt.show()


# 开始预测
# 修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。
with fluid.scope_guard(inference_scope):
    # 从指定目录中加载 推理model(inference model)
    [inference_program,  # 推理的program
     feed_target_names,  # 需要在推理program中提供数据的变量名称
     fetch_targets] = fluid.io.load_inference_model(  # fetch_targets: 推断结果
        model_save_dir,  # model_save_dir:模型训练路径
        infer_exe)  # infer_exe: 预测用executor
    # 获取预测数据  #获取uci_housing的测试数据
    infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),
                                batch_size=200)  # 从测试数据中读取一个大小为200的batch数据
    # 从test_reader中分割x
    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype("float32")
    test_y = np.array([data[1] for data in test_data]).astype("float32")
    results = infer_exe.run(inference_program,  # 预测模型
                            feed={feed_target_names[0]: np.array(test_x)},  # 喂入要预测的x值
                            fetch_list=fetch_targets)  # 得到推测结果

    print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val))
        infer_results.append(val)
    print("ground truth:")
    for idx, val in enumerate(test_y):
        print("%d: %.2f" % (idx, val))
        groud_truths.append(val)
    draw_infer_result(groud_truths, infer_results)