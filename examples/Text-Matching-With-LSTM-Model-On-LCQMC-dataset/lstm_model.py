import paddle.fluid as fluid
#使用飞桨实现一个长短时记忆模型
class SimpleLSTMRNN(fluid.Layer):
    
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None):
        
        #这个模型有几个参数：
        #1. hidden_size，表示embedding-size，或者是记忆向量的维度
        #2. num_steps，表示这个长短时记忆网络，最多可以考虑多长的时间序列
        #3. num_layers，表示这个长短时记忆网络内部有多少层，我们知道，
        # 给定一个形状为[batch_size, seq_len, embedding_size]的输入，
        # 长短时记忆网络会输出一个同样为[batch_size, seq_len, embedding_size]的输出，
        # 我们可以把这个输出再链到一个新的长短时记忆网络上
        # 如此叠加多层长短时记忆网络，有助于学习更复杂的句子甚至是篇章。
        #4. init_scale，表示网络内部的参数的初始化范围，
        # 长短时记忆网络内部用了很多tanh，sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果，
        
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        # weight_1_arr用于存储不同层的长短时记忆网络中，不同门的W参数
        self.weight_1_arr = []
        self.weight_2_arr = []
        # bias_arr用于存储不同层的长短时记忆网络中，不同门的b参数
        self.bias_arr = []
        self.mask_array = []

        # 通过使用create_parameter函数，创建不同长短时记忆网络层中的参数
        # 通过上面的公式，我们知道，我们总共需要8个形状为[_hidden_size, _hidden_size]的W向量
        # 和4个形状为[_hidden_size]的b向量，因此，我们在声明参数的时候，
        # 一次性声明一个大小为[self._hidden_size * 2, self._hidden_size * 4]的参数
        # 和一个 大小为[self._hidden_size * 4]的参数，这样做的好处是，
        # 可以使用一次矩阵计算，同时计算8个不同的矩阵乘法
        # 以便加快计算速度
        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    # 定义LSTM网络的前向计算逻辑，飞桨会自动根据前向计算结果，给出反向结果
    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []
        
        #输入有三个信号：
        # 1. input_embedding，这个就是输入句子的embedding表示，
        # 是一个形状为[batch_size, seq_len, embedding_size]的张量
        # 2. init_hidden，这个表示LSTM中每一层的初始h的值，有时候，
        # 我们需要显示地指定这个值，在不需要的时候，就可以把这个值设置为空
        # 3. init_cell，这个表示LSTM中每一层的初始c的值，有时候，
        # 我们需要显示地指定这个值，在不需要的时候，就可以把这个值设置为空

        # 我们需要通过slice操作，把每一层的初始hidden和cell值拿出来，
        # 并存储在cell_array和hidden_array中
        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        # res记录了LSTM中每一层的输出结果（hidden）
        res = []
        for index in range(self._num_steps):
            # 首先需要通过slice函数，拿到输入tensor input_embedding中当前位置的词的向量表示
            # 并把这个词的向量表示转换为一个大小为 [batch_size, embedding_size]的张量
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])
            
            # 计算每一层的结果，从下而上
            for k in range(self._num_layers):
                # 首先获取每一层LSTM对应上一个时间步的hidden，cell，以及当前层的W和b参数
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                # 我们把hidden和拿到的当前步的input拼接在一起，便于后续计算
                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                
                # 将输入门，遗忘门，输出门等对应的W参数，和输入input和pre-hidden相乘
                # 我们通过一步计算，就同时完成了8个不同的矩阵运算，提高了运算效率
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                # 将b参数也加入到前面的运算结果中
                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                
                # 通过split函数，将每个门得到的结果拿出来
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                
                # 把输入门，遗忘门，输出门等对应的权重作用在当前输入input和pre-hidden上
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                
                # 记录当前步骤的计算结果，
                # m是当前步骤需要输出的hidden
                # c是当前步骤需要输出的cell
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m
                
                # 一般来说，我们有时候会在LSTM的结果结果内加入dropout操作
                # 这样会提高模型的训练鲁棒性
                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
                    
            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))
        
        # 计算长短时记忆网络的结果返回回来，包括：
        # 1. real_res：每个时间步上不同层的hidden结果
        # 2. last_hidden：最后一个时间步中，每一层的hidden的结果，
        # 形状为：[batch_size, num_layers, hidden_size]
        # 3. last_cell：最后一个时间步中，每一层的cell的结果，
        # 形状为：[batch_size, num_layers, hidden_size]
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        
        return real_res, last_hidden, last_cell 
