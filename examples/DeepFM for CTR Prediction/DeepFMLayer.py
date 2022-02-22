import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class DeepFMLayer(nn.Layer):
    # DeepFMLayer由FM和DNN两层组成
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes):
        super(DeepFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes
        # 加载FM和DNN两部分
        self.fm = FM(sparse_feature_number, sparse_feature_dim,
                     dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                       dense_feature_dim, dense_feature_dim + sparse_num_field,
                       layer_sizes)
        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    # 前向传播预测
    def forward(self, sparse_inputs, dense_inputs):
        y_first_order, y_second_order, feat_embeddings = self.fm(sparse_inputs,
                                                                 dense_inputs)
        y_dnn = self.dnn(feat_embeddings)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FM(nn.Layer):
    # FM层，负责抽取low-order特征
    def __init__(self,
                 sparse_feature_number=1000001,
                 sparse_feature_dim=9,
                 dense_feature_dim=13,
                 sparse_num_field=26):
        super(FM, self).__init__()
        self.sparse_feature_number = sparse_feature_number  # 1000001
        self.sparse_feature_dim = sparse_feature_dim  # 9
        self.dense_feature_dim = dense_feature_dim  # 13
        self.sparse_num_field = sparse_num_field  # sparse_inputs_slots-1==>26
        self.layer_sizes = layer_sizes  # fc_sizes: [512, 256, 128, 32]

        # 一阶稀疏特征
        self.sparse_feature_oneOrderWeight = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            padding_idx=0,
            sparse=True
        )
        ## 一阶连续特征
        self.dense_feature_oneOrderWeight = paddle.create_parameter(
            [dense_feature_dim],
            "float32"
        )
        # 二阶特征
        self.sparse_latent_vecs = paddle.nn.Embedding(
            sparse_feature_number,
            embedding_dim,
            padding_idx=0,
            sparse=True
        )
        self.dense_latent_vecs = paddle.create_parameter(
            [1, dense_feature_dim, embedding_dim],
            "float32"
        )

    def forward(self, sparse_feature, dense_feature):
        # 一阶特征

        '''
        计算一阶特征: y_1order = 0 + w*x
        input [batchsize,field_num]
        embed [batchsize,field_num,embedDim]
        sum out axis=1:[batchsize,embedDim]
        '''
        # 稀疏特征查表获得w*x  <- w*1 <- w <- lookup Embedding Table
        sparse_wx = self.sparse_feature_oneOrderWeight(sparse_feature)  # [batchsize,sparse_field_num,1]
        # 连续特征向量内积w*x
        dense_wx = paddle.multiply(dense_feature, self.dense_feature_oneOrderWeight)  # [batchsize,dense_feature_dim]
        dense_wx = paddle.unsqueeze(dense_wx, axis=2)  # [batchsize,dense_feature_dim,1]

        y_pred_first_order = paddle.sum(sparse_wx, axis=1) + paddle.sum(dense_wx,
                                                                        axis=1)  # [batchsize,dense_feature_dim,1]---> [batchsize,1]

        # 二阶特征交叉
        '''
        y_2order=\sum{<Vi,Vj>xi xj}
        优化后计算公式为：
        vi,j * xi的平方和 减去 vi,j * vi 的和的平方，再取1/2   
        '''
        # 稀疏特征查表: vij*xi<-vij *1
        sparse_vx = self.sparse_latent_vecs(sparse_feature)  # [batchsize,sparse_field_num,embed_dim]
        '''
        连续特征矩阵乘法：

        dense_fea: [batchsize,dense_fea_dim,1]
        dense_latent_vecs:[1,dense_fea_dim,embed_dim]
        vij*xi <-  广播逐元素乘法（dense_fea，dense_latent_vecs）  #[batchsize,dense_fea_dim,embed_dim]
        '''
        dense_x = paddle.unsqueeze(dense_feature, axis=2)  # [batchsize,dense_fea_dim]->[batchsize,dense_fea_dim,1]
        dense_vx = paddle.multiply(dense_x, self.dense_latent_vecs)  # [batchsize,dense_fea_dim,embed_dim]

        concat_vx = paddle.concat([sparse_vx, dense_vx], axis=1)  # [batchsize,sparse_field_num+dense_fea_dim,embed_dim]
        embedding = concat_vx
        # 平方的和
        concat_vx_square = paddle.square(concat_vx)  # [batchsize,sparse_field_num+dense_fea_dim,embed_dim]
        square_sum = paddle.sum(concat_vx_square, axis=1)  # [batchsize,embed_dim]
        # 和的平方
        concat_vx_sum = paddle.sum(concat_vx, axis=1)  # [batchsize,embed_dim]
        sum_square = paddle.square(concat_vx_sum)  # [batchsize,embed_dim]

        y_pred_second_order = 0.5 * (paddle.sum(sum_square - square_sum, axis=1))  # [batchsize,1]
        y_pred_second_order = paddle.unsqueeze(y_pred_second_order, axis=1)
        return y_pred_first_order, y_pred_second_order, embedding


class DNN(paddle.nn.Layer):
    # DNN层，负责抽取high-order特征
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
        # 利用FM模型的隐特征向量作为网络权重初始化来获得子网络输出向量
        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    # 得到输入层到embedding层该神经元相连的五条线的权重
    # 前向传播反馈
    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn