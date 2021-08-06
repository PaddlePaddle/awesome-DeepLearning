def build_optimizer(cfg, lr_scheduler, parameters=None):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    return OPTIMIZERS.get(name)(lr_scheduler, parameters=parameters, **cfg_)
