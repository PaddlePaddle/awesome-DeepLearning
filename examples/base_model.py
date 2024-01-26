def setup_optimizers(self, lr, cfg):
    if cfg.get('name', None):
        cfg_ = cfg.copy()
        net_names = cfg_.pop('net_names')
        parameters = []
        for net_name in net_names:
            parameters += self.nets[net_name].parameters()
        self.optimizers['optim'] = build_optimizer(cfg_, lr, parameters)
    else:
        for opt_name, opt_cfg in cfg.items():
            cfg_ = opt_cfg.copy()
            net_names = cfg_.pop('net_names')
            parameters = []
            for net_name in net_names:
                parameters += self.nets[net_name].parameters()
            self.optimizers[opt_name] = build_optimizer(
                cfg_, lr, parameters)

    return self.optimizers
