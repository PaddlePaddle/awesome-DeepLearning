import paddle.nn as nn
import paddle.optimizer as optimizer

class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self,
                 gamma=[0.1, 0.01],
                 milestones=[8, 11],
                 values=None,
                 use_warmup=True):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values
        self.use_warmup = use_warmup

    def __call__(self,
                 base_lr=None,
                 boundary=None,
                 value=None,
                 step_per_epoch=None):
        if boundary is not None and self.use_warmup:
            boundary.extend([int(step_per_epoch) * i for i in self.milestones])
        else:
            # do not use LinearWarmup
            boundary = [int(step_per_epoch) * i for i in self.milestones]
            value = [base_lr]  # during step[0, boundary[0]] is base_lr

        # self.values is setted directly in config 
        if self.values is not None:
            assert len(self.milestones) + 1 == len(self.values)
            return optimizer.lr.PiecewiseDecay(boundary, self.values)

        # value is computed by self.gamma
        value = value if value is not None else [base_lr]
        for i in self.gamma:
            value.append(base_lr * i)

        return optimizer.lr.PiecewiseDecay(boundary, value)

class LearningRate(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    def __init__(self,
                 base_lr=0.01,
                 schedulers=[PiecewiseDecay()]):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr
        self.schedulers = schedulers
        
    def __call__(self, step_per_epoch):
        return self.schedulers(base_lr=self.base_lr,
                                    step_per_epoch=step_per_epoch)

class OptimizerBuilder():
    """
    Build optimizer handles
    Args:
        regularizer (object): an `Regularizer` instance
        optimizers (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 clip_grad_by_norm=None,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizers={'type': 'Momentum',
                            'momentum': .9}):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.regularizer = regularizer
        self.optimizers = optimizers

    def __call__(self, learning_rate, params=None):
        if self.clip_grad_by_norm is not None:
            grad_clip = nn.ClipGradByGlobalNorm(
                clip_norm=self.clip_grad_by_norm)
        else:
            grad_clip = None
        if self.regularizer and self.regularizer != 'None':
            reg_type = self.regularizer['type'] + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, reg_type)(reg_factor)
        else:
            regularization = None

        optim_args = self.optimizers.copy()
        optim_type = optim_args['type']
        del optim_args['type']
        if optim_type != 'AdamW':
            optim_args['weight_decay'] = regularization
        op = getattr(optimizer, optim_type)
        return op(learning_rate=learning_rate,
                  parameters=params,
                  grad_clip=grad_clip,
                  **optim_args)