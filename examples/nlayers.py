class NLayerDiscriminator(nn.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = build_norm_layer(norm_type)
        if type(
                norm_layer
        ) == functools.partial:  # batchnorm2d具有仿射参数，因此无需使用偏差
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D

        kw = 4
        padw = 1

        if norm_type == 'spectral':
            sequence = [
                Spectralnorm(
                    nn.Conv2D(input_nc,
                              ndf,
                              kernel_size=kw,
                              stride=2,
                              padding=padw)),
                nn.LeakyReLU(0.01)
            ]
        else:
            sequence = [
                nn.Conv2D(input_nc,
                          ndf,
                          kernel_size=kw,
                          stride=2,
                          padding=padw,
                          bias_attr=use_bias),
                nn.LeakyReLU(0.2)
            ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐渐增加过滤器的数量
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if norm_type == 'spectral':
                sequence += [
                    Spectralnorm(
                        nn.Conv2D(ndf * nf_mult_prev,
                                  ndf * nf_mult,
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw)),
                    nn.LeakyReLU(0.01)
                ]
            else:
                sequence += [
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw,
                              bias_attr=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        if norm_type == 'spectral':
            sequence += [
                Spectralnorm(
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=1,
                              padding=padw)),
                nn.LeakyReLU(0.01)
            ]
        else:
            sequence += [
                nn.Conv2D(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=kw,
                          stride=1,
                          padding=padw,
                          bias_attr=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        if norm_type == 'spectral':
            sequence += [
                Spectralnorm(
                    nn.Conv2D(ndf * nf_mult,
                              1,
                              kernel_size=kw,
                              stride=1,
                              padding=padw,
                              bias_attr=False))
            ]  # 输出1通道预测图
        else:
            sequence += [
                nn.Conv2D(ndf * nf_mult,
                          1,
                          kernel_size=kw,
                          stride=1,
                          padding=padw)
            ]  # 输出1通道预测图

        self.model = nn.Sequential(*sequence)
        self.final_act = F.sigmoid if use_sigmoid else (lambda x:x)

    def forward(self, input):
        """Standard forward."""
        return self.final_act(self.model(input))
