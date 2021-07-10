#!/usr/bin/env python
# coding: utf-8

# In[2]:


import paddle.fluid as fluid
import paddle
from paddle.dataset import uci_housing
import numpy as np


# In[4]:


paddle.enable_static()
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
hidden = fluid.layers.fc(input=x, size=100, act='relu')
y_predict = fluid.layers.fc(input=hidden, size=1, act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

train_reader = fluid.io.batch(reader=uci_housing.train(),batch_size=128)

feeder = fluid.DataFeeder(place=place, feed_list=[x, y])


# In[5]:


for pass_id in range(1000):
    train_cost = 0
    for data in train_reader():
        train_cost = exe.run(program = fluid.default_main_program(),
                            feed = feeder.feed(data),
                            fetch_list = [avg_cost])
    if pass_id % 100 == 0:
        print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0],))

params_dirname = "result"
fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)


# In[7]:



infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets
        ] = fluid.io.load_inference_model(params_dirname, infer_exe)
    batch_size = 10

    infer_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)

    infer_data = next(infer_reader())
    infer_feat = np.array(
            [data[0] for data in infer_data]).astype("float32")
    
    infer_label = np.array(
            [data[1] for data in infer_data]).astype("float32")

    
    assert feed_target_names[0] == 'x'
    
    results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: np.array(infer_feat)},
            fetch_list=fetch_targets)

    
    print("infer results: (House Price)")
    
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val))

    print("\nground truth:")
    
    for idx, val in enumerate(infer_label):
        print("%d: %.2f" % (idx, val))

