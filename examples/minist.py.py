
'''
残差网络结构resnet

resnet的网络结构,它对每层的输入做一个reference, 学习形成残差函数， 而不是学习一些没有reference的函数。这种残差函数更容易优化，能使网络层数大大加深。

我们知道，在计算机视觉里，特征的“等级”随增网络深度的加深而变高，研究表明，网络的深度是实现好的效果的重要因素。然而梯度弥散/爆炸成为训练深层次的网络的障碍，导致无法收敛。
有一些方法可以弥补，如归一初始化，各层输入归一化，使得可以收敛的网络的深度提升为原来的十倍。然而，虽然收敛了，但网络却开始退化了，即增加网络层数却导致更大的误差。

的确，通过在一个浅层网络基础上叠加y=x的层（称identity mappings，恒等映射），可以让网络随深度增加而不退化。这反映了多层非线性网络无法逼近恒等映射网络。

但是，不退化不是我们的目的，我们希望有更好性能的网络。  resnet学习的是残差函数F(x) = H(x) - x, 这里如果F(x) = 0, 那么就是上面提到的恒等映射。事实上，resnet是“shortcut connections”
的在connections是在恒等映射下的特殊情况，它没有引入额外的参数和计算复杂度。
假如优化目标函数是逼近一个恒等映射, 而不是0映射， 那么学习找到对恒等映射的扰动会比重新学习一个映射函数要容易。残差函数一般会有较小的响应波动，表明恒等映射是一个合理的预处理。

通过一个shortcut，和第2个ReLU，获得输出y。当需要对输入和输出维数进行变化时（如改变通道数目），可以在shortcut时对x做一个线性变换Ws

实验证明，这个残差块往往需要两层以上，单单一层的残差块(y=W1x+x)并不能起到提升作用。

残差网络的确解决了退化的问题，在训练集和校验集上，都证明了的更深的网络错误率越小。

实际中，考虑计算的成本，对残差块做了计算优化，即将两个3x3的卷积层替换为1x1 + 3x3 + 1x1,新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。

本次resnet用于手写体识别中构建了一个17层的网络，在设计网络结构的时候，加上dropout处理。并进行进行batch normalization处理，这样能在一定情况下让数据更有效。
'''

import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

# 准备数据
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))

# 定义 placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# 定义模型
network = tl.layers.InputLayer(x, name='input_layer')
res_a = network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu1')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu2')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu3')
res_a = network = tl.layers.ElementwiseLayer([network, res_a], combine_fn=tf.add, name='res_add1')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu4')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu5')
res_a = network = tl.layers.ElementwiseLayer([network, res_a], combine_fn=tf.add, name='res_add2')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu6')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu7')
res_a = network = tl.layers.ElementwiseLayer([network, res_a], combine_fn=tf.add, name='res_add3')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu8')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu9')
res_a = network = tl.layers.ElementwiseLayer([network, res_a], combine_fn=tf.add, name='res_add4')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu10')
network = tl.layers.DenseLayer(network, n_units=200, act = tf.nn.elu, name='relu11')
res_a = network = tl.layers.ElementwiseLayer([network, res_a], combine_fn=tf.add, name='res_add5')
network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity, name='output_layer')
# 定义损失函数和衡量指标
# tl.cost.cross_entropy 在内部使用 tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name = 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# 定义 optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.003, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# 初始化 session 中的所有参数
tl.layers.initialize_global_variables(sess)

# 列出模型信息
network.print_params()
network.print_layers()

# 训练模型
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# 评估模型
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# 把模型保存成 .npz 文件
tl.files.save_npz(network.all_params , name='model.npz')
sess.close()
