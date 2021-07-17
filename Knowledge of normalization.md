一、归一化--层归一化
   1、概念
         如果一个神经元的净输入分布在神经网络中是动态变化的，比如循环神经网络，那么无法应用批归一化操作。
层归一化和批归一化不同的是，层归一化是对一个中间层的所有神经元进行归一化。
        局部响应归一化（Local Response Normalization），简称LRN，是在深度学习中用于提高准确度，且一般会在激活、池化后进行的一种处理方法，通常应用于Alexnet中。
        归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保证程序运行时收敛加快。归一化的具体作用是归纳统一样本的统计分布性。归一化在0-1之间是统计的概率分布，归一化在某个区间上是统计的坐标分布。归一化有同一、统一和合一的意思。
        归一化的目的简而言之，是使得没有可比性的数据变得具有可比性，同时又保持相比较的两个数据之间的相对关系，如大小关系；或是为了作图，原来很难在一张图上作出来，归一化后就可以很方便的给出图上的相对位置等。
   2.算法流程
         from keras_layer_normalization import LayerNormalization
   # 构建LN CNN网络
   model_ln = Sequential()
   model_ln.add(Conv2D(input_shape = (28,28,1), filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
   model_ln.add(MaxPool2D(pool_size=(2,2), strides=2))
   model_ln.add(Conv2D(input_shape=(14,14,6), filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
   model_ln.add(MaxPool2D(pool_size=(2,2), strides=2))
   model_ln.add(Flatten())
   model_ln.add(Dense(120, activation='tanh'))
   model_ln.add(LayerNormalization()) # 添加LN运算
   model_ln.add(Dense(84, activation='tanh'))
   model_ln.add(LayerNormalization())
   model_ln.add(Dense(10, activation='softmax'))
   from lstm_ln import LSTM_LN

   model_ln = Sequential()
   model_ln.add(Embedding(max_features,100))
   model_ln.add(LSTM_LN(128))
   model_ln.add(Dense(1, activation='sigmoid'))
   model_ln.summary()
   3、作用 
          批量归一化是对一个中间层的单个神经元进行归一化操作，因此要求小批量样本的数量不能太小，
否则难以计算单个神经元的统计信息。层归一化（Layer Normalization）是和批量归一化非常类似的方法。
和批量归一化不同的是，层归一化是对某一层的所有神经元进行归一化。
          在使用机器学习算法的数据预处理阶段，归一化也是非常重要的一个步骤。例如在应用SVM之前，
缩放是非常重要的。Sarle的神经网络FAQ的第二部分（1997）阐述了缩放的重要性，大多数注意事项也适用于SVM。
缩放的最主要优点是能够避免大数值区间的属性过分支配了小数值区间的属性。另一个优点能避免计算过程中数值复杂度。
因为关键值通常依赖特征向量的内积（inner products），例如，线性核和多项式核，属性的大数值可能会导致数值问题。
我们推荐将每个属性线性缩放到区间[-1,+1]或者[0, 1]。