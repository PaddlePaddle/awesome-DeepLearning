class LinearDecay(LambdaDecay):
    def __init__(self, learning_rate, start_epoch, decay_epochs,
                 iters_per_epoch):
        def lambda_rule(epoch):
            epoch = epoch // iters_per_epoch
            lr_l = 1.0 - max(0,
                             epoch + 1 - start_epoch) / float(decay_epochs + 1)
            return lr_l

        super().__init__(learning_rate, lambda_rule)
