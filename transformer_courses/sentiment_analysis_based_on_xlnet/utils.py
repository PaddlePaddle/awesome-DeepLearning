import paddle
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

class Config():
    def __init__(self):
        self.task_name = "sst-2"
        self.model_name_or_path = "xlnet-base-cased"
        self.output_dir = "./tmp"
        self.max_seq_length = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 3
        self.max_steps = -1
        self.logging_steps = 100
        self.save_steps=500
        self.seed=43
        self.device="gpu"
        self.warmup_steps = 0
        self.warmup_proportion = 0.1
