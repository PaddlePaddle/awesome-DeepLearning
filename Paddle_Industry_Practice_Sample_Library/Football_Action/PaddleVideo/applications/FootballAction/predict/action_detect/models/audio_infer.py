"""
ppTSM InferModel
"""
import sys
import numpy as np
import time

sys.path.append('../')
from utils.preprocess import get_images
from utils.config_utils import parse_config

import reader
from paddle.inference import Config
from paddle.inference import create_predictor


class InferModel(object):
    """audio infer"""
    def __init__(self, cfg, name='AUDIO'): 
        name = name.upper()
        self.name           = name
        model_file          = cfg[name]['model_file']
        params_file         = cfg[name]['params_file']
        gpu_mem             = cfg[name]['gpu_mem']
        device_id           = cfg[name]['device_id']

        # model init
        config = Config(model_file, params_file)
        config.enable_use_gpu(gpu_mem, device_id)
        config.switch_ir_optim(True)  # default true
        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        self.predictor = create_predictor(config)

        input_names = self.predictor.get_input_names()
        self.input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        self.output_tensor = self.predictor.get_output_handle(output_names[0])


    def infer(self, input):
        """infer"""
        self.input_tensor.copy_from_cpu(input)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


    def predict(self, infer_config):
        """predict"""
        infer_reader = reader.get_reader(self.name, 'infer', infer_config)
        feature_list = []
        pcm_list = []
        for infer_iter, data in enumerate(infer_reader()):
            inputs = np.array(data, dtype = 'float32')
            output = self.infer(inputs)
            feature_list.append(np.squeeze(output))
            pcm_list.append(inputs)
        feature_values = np.vstack(feature_list)
        pcm_values = np.vstack(pcm_list)
        return feature_values, pcm_values


if __name__ == "__main__":
    cfg_file = '/home/work/inference/configs/configs.yaml' 
    cfg = parse_config(cfg_file)
    model = InferModel(cfg)

    pcm_path = '/home/work/datasets/WorldCup2018/pcm/6e577252c4004961ac7caa738a52c238.pcm'
    t0 = time.time()
    cfg['AUDIO']['pcm_file'] = pcm_path
    outputs = model.predict(cfg)
    # outputs = model.infer(np.random.rand(32, 8, 3, 224, 224).astype(np.float32))
    t1 = time.time()

    print(outputs.shape)
    print(outputs[0])
    print('cost time = {} min'.format((t1 - t0) / 60.0))
