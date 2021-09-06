# save
#paddle.save(deepFM_model.state_dict(), "./model/deepFM_model.pdparams")
#paddle.save(adam.state_dict(), "./model/adam.pdopt")

import paddle

layer_state_dict = paddle.load("./model/deepFM_model.pdparams")
opt_state_dict = paddle.load("./model/adam.pdopt")

testDeepFM_model=DeepFMLayer(sparse_feature_number = 1000001, sparse_feature_dim = 9,
                 dense_feature_dim = 13, sparse_num_field = 26, layer_sizes = [512, 256, 128, 32])
testAdam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=testDeepFM_model.parameters())# Adam优化器

testDeepFM_model.set_state_dict(layer_state_dict)
testAdam.set_state_dict(opt_state_dict)