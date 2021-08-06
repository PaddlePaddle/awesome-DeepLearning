class CycleGANModel(BaseModel):
    def __init__(self,
                 generator,
                 discriminator=None,
                 cycle_criterion=None,
                 idt_criterion=None,
                 gan_criterion=None,
                 pool_size=50,
                 direction='a2b',
                 lambda_a=10.,
                 lambda_b=10.):
        super(CycleGANModel, self).__init__()
        self.direction = direction

        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        # 定义生成器，和论文中命名有所区别
        self.nets['netG_A'] = build_generator(generator)
        self.nets['netG_B'] = build_generator(generator)
        init_weights(self.nets['netG_A'])
        init_weights(self.nets['netG_B'])

        # 定义鉴别器
        if discriminator:
            self.nets['netD_A'] = build_discriminator(discriminator)
            self.nets['netD_B'] = build_discriminator(discriminator)
            init_weights(self.nets['netD_A'])
            init_weights(self.nets['netD_B'])

        # 创建图片流存储之前生成的图片
        self.fake_A_pool = ImagePool(pool_size) 
        # 创建图片流存储之前生成的图片
        self.fake_B_pool = ImagePool(pool_size)

        # 定义损失函数
        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion) 

        if cycle_criterion:
            self.cycle_criterion = build_criterion(cycle_criterion)

        if idt_criterion:
            self.idt_criterion = build_criterion(idt_criterion)

    def setup_input(self, input):
        AtoB = self.direction == 'a2b'

        if AtoB:
            if 'A' in input:
                self.real_A = paddle.to_tensor(input['A'])
            if 'B' in input:
                self.real_B = paddle.to_tensor(input['B'])
        else:
            if 'B' in input:
                self.real_A = paddle.to_tensor(input['B'])
            if 'A' in input:
                self.real_B = paddle.to_tensor(input['A'])

        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        elif 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self):
        # 前向传播
        if hasattr(self, 'real_A'):
            self.fake_B = self.nets['netG_A'](self.real_A)  # G_A(A)
            self.rec_A = self.nets['netG_B'](self.fake_B)  # G_B(G_A(A))

            # 可视化
            self.visual_items['real_A'] = self.real_A
            self.visual_items['fake_B'] = self.fake_B
            self.visual_items['rec_A'] = self.rec_A

        if hasattr(self, 'real_B'):
            self.fake_A = self.nets['netG_B'](self.real_B)  # G_B(B)
            self.rec_B = self.nets['netG_A'](self.fake_A)  # G_A(G_B(B))

            # 可视化
            self.visual_items['real_B'] = self.real_B
            self.visual_items['fake_A'] = self.fake_A
            self.visual_items['rec_B'] = self.rec_B

    def backward_D_basic(self, netD, real, fake):
    # 真
    pred_real = netD(real)
    loss_D_real = self.gan_criterion(pred_real, True)
    # 假
    pred_fake = netD(fake.detach())
    loss_D_fake = self.gan_criterion(pred_fake, False)
    # 合并损失并计算梯度
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    loss_D.backward()
    return loss_D

    def backward_D_A(self):
    # 计算鉴别器D_A的GAN损失
    fake_B = self.fake_B_pool.query(self.fake_B)
    self.loss_D_A = self.backward_D_basic(self.nets['netD_A'], self.real_B,
                                            fake_B)
    self.losses['D_A_loss'] = self.loss_D_A
    
    def backward_D_B(self):
    # 计算鉴别器D_B的GAN损失
    fake_A = self.fake_A_pool.query(self.fake_A)
    self.loss_D_B = self.backward_D_basic(self.nets['netD_B'], self.real_A,
                                            fake_A)
    self.losses['D_B_loss'] = self.loss_D_B

    def backward_G(self):
    """Calculate the loss for generators G_A and G_B"""
    # 计算生成器G_A和G_B的损失
    # Identity损失
    if self.idt_criterion:
        self.idt_A = self.nets['netG_A'](self.real_B)

        self.loss_idt_A = self.idt_criterion(self.idt_A,
                                                self.real_B) * self.lambda_b
        self.idt_B = self.nets['netG_B'](self.real_A)

        # 可视化
        self.visual_items['idt_A'] = self.idt_A
        self.visual_items['idt_B'] = self.idt_B

        self.loss_idt_B = self.idt_criterion(self.idt_B,
                                                self.real_A) * self.lambda_a
    else:
        self.loss_idt_A = 0
        self.loss_idt_B = 0
    # GAN损失D_A(G_A(A))
    self.loss_G_A = self.gan_criterion(self.nets['netD_A'](self.fake_B),
                                        True)
    # GAN损失D_B(G_B(B))
    self.loss_G_B = self.gan_criterion(self.nets['netD_B'](self.fake_A),
                                        True)
    # 前向循环损失|| G_B(G_A(A)) - A||
    self.loss_cycle_A = self.cycle_criterion(self.rec_A,
                                                self.real_A) * self.lambda_a
    # 反向循环损失|| G_A(G_B(B)) - B||
    self.loss_cycle_B = self.cycle_criterion(self.rec_B,
                                                self.real_B) * self.lambda_b

    self.losses['G_idt_A_loss'] = self.loss_idt_A
    self.losses['G_idt_B_loss'] = self.loss_idt_B
    self.losses['G_A_adv_loss'] = self.loss_G_A
    self.losses['G_B_adv_loss'] = self.loss_G_B
    self.losses['G_A_cycle_loss'] = self.loss_cycle_A
    self.losses['G_B_cycle_loss'] = self.loss_cycle_B
    # 合并损失并计算梯度
    self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

    self.loss_G.backward()   
