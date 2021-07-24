# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class DeepFM(BaseEstimator, TransformerMixin):
    '''

         很简单的道理 就是   Deep + FM 的结合    （这里没考虑域，在另一个模型DeepFFM才开始考虑到域和隐向量，不要混杂，另一个是deep+FM，）
         其中 FM 用于低阶特征提取
              Deep用于高阶特征提取。

         共享相同的嵌入层输入。对field做embeding嵌入， 嵌入结果  作为FM部分和  dnn部分的输入。


         所以首先第一步就是做了embedidng操作，其本质是一层全连接的神经网络，来将特征转成向量。

         可参考此处的解释，代码解释挺多的：https://blog.csdn.net/qq_15111861/article/details/94194240

    '''
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result,self.valid_result = [],[]

        self._init_graph()
        print('feature_size:', self.feature_size)
        print('field_size:',self.field_size)
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            '''
            主要的图定义的部分，可以主要参考着网络架构来看
            
            feat_index：特征的一个序号，主要用于通过embedding_lookup选择我们的embedding。 （单纯的特征的embedding转换）
            feat_value：对应的特征值，如果是离散特征的话，就是1，如果不是离散特征的话，就保留原来的特征值。
            label：实际值。还定义了两个dropout来防止过拟合。
            
            
            这里整体的结果 有个疑问是对 不同的  field 怎么进行操作的，   deep 和 FFM部分看图是有不同输入组织的...好吧deepFM内还没考虑域
            
            
            在这之后，调用权重的初始化方法。将所有的权重放到一个字典中。feature_embeddings 本质上就是 FM 中的 latent vector 。
            对于每一个特征都建立一个隐特征向量。feature_bias 代表了 FM 中的 w 的权重。然后就是搭建深度图，输入到深度网络的大小
            为：特征的个数 * 每个隐特征向量的长度。根据每层的配置文件，生产相应的权重。对于输出层，根据不同的配置，生成不同的
            输出的大小。如果只是使用 FM 算法，那么
            
            
            
            '''

            tf.set_random_seed(self.random_seed)


            # 分别是特征 对应特征字典的索引 、  对应特征值
            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None,None],
                                             name='feat_index')
            self.feat_value = tf.placeholder(tf.float32,
                                           shape=[None,None],
                                           name='feat_value')

            self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_fm = tf.placeholder(tf.float32,shape=[None],name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool,name='train_phase')

            self.weights = self._initialize_weights()


            print('-------   嵌入层，用于根据特征索引获取特征对应的embedding   ---------')
            # model     模型的第一部分都是做embeding,对应就是此部分   共有的 embedding部分
            #根据feat_index选择对应的weights['feature_embeddings']中的embedding值，然后再与对应的feat_value相乘就可以了
            #对应公式中的  隐向量v,参考此处的解释：https://www.jianshu.com/p/cf796fca244c
            '''
            根据每次输入的特征的索引，从隐特征向量中取出其对应的隐向量。将每一个特征对应的具体的值，和自己对应的隐向量相乘。
            如果是 numerical 的，就直接用对应的 value 乘以隐向量。如果是 categories 的特征，其对应的特征值是 1，相乘完还是原来的隐向量。
            最后，self.embeddings 存放的就是输入的样本的特征值和隐向量的乘积。大小为 batch_sizefield_sizeembedding_size
            '''
            ###########   这里N * F * K 表示       特征数量 * Field隐向量数量  *隐向量长度
            '''
                 要求到的是 每个特征 针对其他field的embeddings.
                 (1)从weights['feature_embeddings'] 中根据特征索引找到  自己这个样本包含的特征的所有隐变量
                （2）对样本对应的特征值 进行reshape一下，维度是 -1表示不明确尺寸，之后是确定 field_size行一列。三维的
                （3）以上两步的结果进行相乘，结果embeddings的含义是什么呢，对feature对应的embeding特定feature值 和 特征值相乘，
                
            根据此处https://www.jianshu.com/p/e7b2d53ec42b 对嵌入层的说明，不同field的长度 经过embedding之后向量的长度均为embedding-size。
            
            可算是明白了，这里有一个很大很大的弯，就是feat_value 部分为什么要是field_size，这是因为经过one-hot过的有效的列值是跟field是
            对应的，因为我们针对一个离散field转成 one-hot表达时候，其实也只保存了一个为1的特征，而其他无效的是没保存的，所以 feat_value的长度尺寸其实就是field_size,
            这里就是  第一步的维度是field* embedding-size的， 第二步提取出每个field下的value,  第三步相乘，相当于获取了对每个field_value的embedding。
            这个embeding相对于从 feature_size尺寸到embedding-size对每个特征的降维，维护的是全长的 one-hot后的特征size。
             
            
            难点就是把握好    field_size  和  embedding-size  以及   feature_size 的含义和区别。最后虽然特征索引长度是feature_size，但是真实有效存储的列表长度还是field_size。
            而embedidng时候的权值矩阵应为可以按索引取，对应的行数还是feature_size。  最后维度就是field_size * self.embedding_size 表示特征与v的相乘
            '''
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            print('self.embeddings.shape:',self.embeddings.shape)
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            print('feat_value.shape:', self.embeddings.shape)
            self.embeddings = tf.multiply(self.embeddings,feat_value)
            print('after self.embeddings.shape:', self.embeddings.shape)



            print('---------     第一部分网络的定义，FM的一次项计算部分（第一、二合并整体是FM部分）      -----------')

            '''
            计算一阶项，从 self.weights[“feature_bias”] 取出对应的 w ，得到一阶项，大小为 batch_size*field_size。
二阶         二阶项的计算，也就是 FM 的计算，利用了的技巧。先将 embeddings 在 filed_size 的维度上求和，最后得到红框里面的项
            '''

            # first order term        代表FM公式中的一次项     这里的feat_index 是用来去除其对应的权值的，
            #这里只经过 简单的  weights ，进行了简单的w*x处理， 做了对应feature下的单值相乘
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'],self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order,feat_value),2)
            self.y_first_order = tf.nn.dropout(self.y_first_order,self.dropout_keep_fm[0])

            print('---------     FM的二次项计算部分（对应于整体的公式后部分）      -----------')
            # second order term         这整体区间代表FM公式中的二次项计算
            # sum-square-part   在公式中 相减的前一部分。  先加后平方   这里的1表示维度内相加，对应公式中的，对所有的u*x的结果相加
            self.summed_features_emb = tf.reduce_sum(self.embeddings,1) # None * k
            self.summed_features_emb_square = tf.square(self.summed_features_emb) # None * K

            # squre-sum-part  在公式中 相减的后一部分。  先平方后加
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            #second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order,self.dropout_keep_fm[1])

            print('---------     第二部分网络的定义，deep部分深度网络结构的定义      -----------')
            '''
            计算 deep 的项。将 self.embeddings(大小为 batch_sizeself.field_size * self.embedding_size) reshape 
            成 batch_size(self.field_size * self.embedding_size) 的大小，然后输入到网络里面进行计算。
            '''
            # Deep component     将Embedding part的输出再经过几层全链接层
            self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])

            for i in range(0,len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["layer_%d" %i]), self.weights["bias_%d"%i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])



            ######################## 这里是一个特色吧，之前相当于把所有的结构都定义过了，这里会决定对那些结构进行使用，组成不同的网络。
            #----DeepFM--------- 我们可以使用logloss(如果定义为分类问题)，或者mse(如果定义为预测问题)，以及多种的优化器去进行尝试
            '''
            最后将所有项 concat 起来，投影到一个值。如果是只要 FM ，不要 deep 的部分，则投影的大小为 filed_size+embedding_size 的大小。
            如果需要 deep 的部分，则大小再加上 deep 的部分。利用最后的全连接层，将特征映射到一个 scalar 
            '''
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep

            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])


            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)


            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)





    def _initialize_weights(self):
        '''
        对网络参数的初始定义，   这里feature_embeddings  表示的是  隐函数数量的设置。


       weights['feature_embeddings'] ：存放的每一个值其实就是FM中的vik，所以它是F * K的。其中，F代表feture的大小(将离散特征转换成one-hot之后的特征总量),K代表dense vector的大小。
       weights['feature_bias']       ：FM中的一次项的权重

        '''
        weights = dict()

        #embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size,self.embedding_size],0.0,0.01),
            name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size,1],0.0,1.0),name='feature_bias')

        #deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(input_size,self.deep_layers[0])),dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32
        )


        for i in range(1,num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]


        # final concat projection layer
        # 为什么self.field_size + self.embedding_size 是FM输入层的size？

        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)


        return weights


    def get_batch(self,Xi,Xv,y,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end],Xv[start:end],[[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        #通过设置相同的state,使得random.shuffle以相同的规律打乱两个列表，进而使得两个列表被打乱后，仍旧能维持两个列表间元素的一一对应关系
        #为了保持shuffle以后 feature的索引和  对应的值是对应的
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample


        使用模型根据   输入特征情况得到预测结果
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def fit_on_batch(self,Xi,Xv,y):
        #如下是每轮的训练过程
        feed_dict = {self.feat_index:Xi,
                     self.feat_value:Xv,
                     self.label:y,
                     self.dropout_keep_fm:self.dropout_fm,
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None


        主要的训练过程， 做epoch  批次的训练，每次训练都是调用函数fit_on_batch   启动每个批次的网络训练过程。
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            #执行多个 total_batch 的训练
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)


            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break


    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False













