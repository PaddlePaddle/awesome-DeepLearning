"""
ppTSM InferModel
"""
import sys
import numpy as np
import json
import pickle
import time

sys.path.append('../')
from utils.preprocess import get_images
from utils.config_utils import parse_config
from utils.process_result import process_proposal

import reader
from paddle.inference import Config
from paddle.inference import create_predictor


class InferModel(object):
    """bmn infer"""
    def __init__(self, cfg, name='BMN'): 
        name = name.upper()
        self.name           = name
        model_file          = cfg[name]['model_file']
        params_file         = cfg[name]['params_file']
        gpu_mem             = cfg[name]['gpu_mem']
        device_id           = cfg[name]['device_id']

        self.nms_thread          = cfg[name]['nms_thread']
        self.min_pred_score      = cfg[name]['score_thread']
        self.min_frame_thread    = cfg['COMMON']['fps']

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
        self.output1_tensor = self.predictor.get_output_handle(output_names[0])
        self.output2_tensor = self.predictor.get_output_handle(output_names[1])
        self.output3_tensor = self.predictor.get_output_handle(output_names[2])


    def infer(self, input):
        """infer"""
        self.input_tensor.copy_from_cpu(input)
        self.predictor.run()
        output1 = self.output1_tensor.copy_to_cpu()
        output2 = self.output2_tensor.copy_to_cpu()
        output3 = self.output3_tensor.copy_to_cpu()
        return output1, output2, output3


    def generate_props(self, pred_bmn, pred_start, pred_end, max_window=200, min_window=5):
        """generate_props"""
        video_len = min(pred_bmn.shape[-1], min(pred_start.shape[-1], pred_end.shape[-1]))
        pred_bmn = pred_bmn[0, :, :] * pred_bmn[1, :, :]
        start_mask = self.boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = self.boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_results = []
        for idx in range(min_window, max_window):
            for jdx in range(video_len):
                start_index = jdx
                end_index = start_index + idx
                if end_index < video_len and start_mask[start_index] == 1 and end_mask[end_index] == 1:
                    xmin = start_index
                    xmax = end_index
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bmn_score = pred_bmn[idx, jdx]
                    conf_score = xmin_score * xmax_score * bmn_score
                    score_results.append([xmin, xmax, conf_score])
        return score_results


    def boundary_choose(self, score_list):
        """boundary_choose"""
        max_score = max(score_list)
        mask_high = (score_list > max_score * 0.5)
        score_list = list(score_list)
        score_middle = np.array([0.0] + score_list + [0.0])
        score_front = np.array([0.0, 0.0] + score_list)
        score_back = np.array(score_list + [0.0, 0.0])
        mask_peak = ((score_middle > score_front) & (score_middle > score_back))
        mask_peak = mask_peak[1:-1]
        mask = (mask_high | mask_peak).astype('float32')
        return mask


    def predict(self, infer_config, material):
        """predict"""
        infer_reader = reader.get_reader(self.name, 'infer', infer_config, material=material)
        feature_list = []
        for infer_iter, data in enumerate(infer_reader()):
            inputs      = [items[0] for items in data]
            winds       = [items[1] for items in data]
            feat_info   = [items[2] for items in data]
            feature_T   = feat_info[0][0]
            feature_N   = feat_info[0][1]

            inputs = np.array(inputs)
            pred_bmn, pred_sta, pred_end = self.infer(inputs)

            if infer_iter == 0:
                sum_pred_bmn = np.zeros((2, feature_N, feature_T))
                sum_pred_sta = np.zeros((feature_T, ))
                sum_pred_end = np.zeros((feature_T, ))
                sum_pred_cnt = np.zeros((feature_T, ))

            for idx, sub_wind in enumerate(winds):
                sum_pred_bmn[:, :, sub_wind[0]: sub_wind[1]] += pred_bmn[idx]
                sum_pred_sta[sub_wind[0]: sub_wind[1]] += pred_sta[idx]
                sum_pred_end[sub_wind[0]: sub_wind[1]] += pred_end[idx]
                sum_pred_cnt[sub_wind[0]: sub_wind[1]] += np.ones((sub_wind[1] - sub_wind[0], ))

        pred_bmn = sum_pred_bmn / sum_pred_cnt
        pred_sta = sum_pred_sta / sum_pred_cnt
        pred_end = sum_pred_end / sum_pred_cnt

        score_result = self.generate_props(pred_bmn, pred_sta, pred_end)
        results = process_proposal(score_result, self.min_frame_thread, self.nms_thread, self.min_pred_score)

        return results


if __name__ == "__main__":
    cfg_file = '/home/work/inference/configs/configs.yaml' 
    cfg = parse_config(cfg_file)
    model = InferModel(cfg)

    imgs_path = '/home/work/datasets/WorldCup2018/frames/6e577252c4004961ac7caa738a52c238'

    # feature
    feature_path = imgs_path.replace("frames", "features") + '.pkl'
    video_features = pickle.load(open(feature_path, 'rb'))

    t0 = time.time()
    outputs = model.predict(cfg, video_features)
    # outputs = model.infer(np.random.rand(32, 8, 3, 224, 224).astype(np.float32))
    t1 = time.time()

    results = {'proposal': outputs}
    with open('results.json', 'w', encoding='utf-8') as f:
       data = json.dumps(results, indent=4, ensure_ascii=False)
       f.write(data) 
    print('cost time = {} min'.format((t1 - t0) / 60.0))
