class ResnetGenerator(nn.Layer):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='instance',
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)     
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D

        model = [
            nn.Pad2D(padding=[3, 3, 3, 3], mode="reflect"),
            nn.Conv2D(input_nc,
                      ngf,
                      kernel_size=7,
                      padding=0,
                      bias_attr=use_bias),
            norm_layer(ngf),
            nn.ReLU()
        ]

        n_downsampling = 2
        # 添加下采样层
        for i in range(n_downsampling): 
            mult = 2**i
            model += [
                nn.Conv2D(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias_attr=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        mult = 2**n_downsampling
        # 添加ResNet块
        for i in range(n_blocks):  

            model += [
                ResnetBlock(ngf * mult,
                            padding_type=padding_type,
                            norm_layer=norm_layer,
                            use_dropout=use_dropout,
                            use_bias=use_bias)
            ]
        
        # 添加上采样层
        for i in range(n_downsampling):  
            mult = 2**(n_downsampling - i)
            model += [
                nn.Conv2DTranspose(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias_attr=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ]
        model += [nn.Pad2D(padding=[3, 3, 3, 3], mode="reflect")]
        model += [nn.Conv2D(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # 前向传播
        return self.model(x)
