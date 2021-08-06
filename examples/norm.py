def build_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm,
            param_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(1.0, 0.02)),
            bias_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0)),
            trainable_statistics=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2D,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(1.0),
                learning_rate=0.0,
                trainable=False),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0),
                                       learning_rate=0.0,
                                       trainable=False))
    elif norm_type == 'spectral':
        norm_layer = functools.partial(Spectralnorm)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer
