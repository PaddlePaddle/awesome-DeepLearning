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
from utils.process_result import get_action_result

import reader
from paddle.inference import Config
from paddle.inference import create_predictor


class InferModel(object):
    """lstm infer"""
    def __init__(self, cfg, name='ACTION'): 
        name = name.upper()
        self.name           = name
        model_file          = cfg[name]['model_file']
        params_file         = cfg[name]['params_file']
        gpu_mem             = cfg[name]['gpu_mem']
        device_id           = cfg[name]['device_id']

        self.topk           = cfg[name]['topk']
        self.frame_offset   = cfg[name]['nms_offset']
        self.nms_thread     = cfg[name]['nms_thread']
        self.cls_thread     = cfg[name]['classify_score_thread']
        self.iou_thread     = cfg[name]['iou_score_thread']

        self.label_map_file = cfg['COMMON']['label_dic']
        self.fps            = cfg['COMMON']['fps']
        self.nms_id         = 5

        # model init
        config = Config(model_file, params_file)
        config.enable_use_gpu(gpu_mem, device_id)
        config.switch_ir_optim(True)  # default true
        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        self.predictor = create_predictor(config)

        input_names = self.predictor.get_input_names()
        self.input1_tensor = self.predictor.get_input_handle(input_names[0])
        self.input2_tensor = self.predictor.get_input_handle(input_names[1])

        output_names = self.predictor.get_output_names()
        self.output1_tensor = self.predictor.get_output_handle(output_names[0])
        self.output2_tensor = self.predictor.get_output_handle(output_names[1])


    def infer(self, input1_arr, input1_lod, input2_arr=None, input2_lod=None):
        """infer"""
        self.input1_tensor.copy_from_cpu(input1_arr)
        self.input1_tensor.set_lod(input1_lod)
        if not input2_arr is None:
            self.input2_tensor.copy_from_cpu(input2_arr)
            self.input2_tensor.set_lod(input2_lod)
        self.predictor.run()
        output1 = self.output1_tensor.copy_to_cpu()
        output2 = self.output2_tensor.copy_to_cpu()
        # print(output.shape)
        return output1, output2

    def pre_process(self, input):
        """pre process"""
        input_arr = []
        input_lod = [0]
        start_lod = 0
        end_lod = 0
        for sub_item in input:
            end_lod = start_lod + len(sub_item)
            input_lod.append(end_lod)
            input_arr.extend(sub_item)
            start_lod = end_lod
        input_arr = np.array(input_arr)
        # print(input_arr.shape)
        # print([input_lod])
        return input_arr, [input_lod]

    def predict(self, infer_config, material):
        """predict"""
        infer_reader = reader.get_reader(self.name, 'infer', infer_config, material=material)
        results = []
        for infer_iter, data in enumerate(infer_reader()):
            video_id = [[items[-2], items[-1]] for items in data]
            input1 = [items[0] for items in data]
            input2 = [items[1] for items in data]
            input1_arr, input1_lod = self.pre_process(input1)
            input2_arr, input2_lod = self.pre_process(input2)
            output1, output2 = self.infer(input1_arr, input1_lod, input2_arr, input2_lod)
            # output1, output2 = self.infer(input1_arr, input1_lod)

            predictions_id = output1 
            predictions_iou = output2
            for i in range(len(predictions_id)):
                topk_inds = predictions_id[i].argsort()[0 - self.topk:]
                topk_inds = topk_inds[::-1]
                preds_id = predictions_id[i][topk_inds]
                preds_iou = predictions_iou[i][0]
                results.append((video_id[i], preds_id.tolist(), topk_inds.tolist(), preds_iou.tolist()))

        predict_result = get_action_result(results, self.label_map_file, self.fps, 
                                           self.cls_thread, self.iou_thread, 
                                           self.nms_id, self.nms_thread, self.frame_offset)
        return predict_result


if __name__ == "__main__":
    cfg_file = '/home/work/inference/configs/configs.yaml' 
    cfg = parse_config(cfg_file)
    model = InferModel(cfg)

    # proposal total
    prop_dict = {}
    for dataset in ['EuroCup2016', 'WorldCup2018']:
        prop_json = '/home/work/datasets/{}/feature_bmn/prop.json'.format(dataset)
        json_data = json.load(open(prop_json, 'r'))
        for item in json_data:
            basename = prop_json.replace('feature_bmn/prop.json', 'mp4')
            basename = basename + '/' + item['video_name'] + '.mp4'
            prop_dict[basename] = item['bmn_results']

    imgs_path = '/home/work/datasets/WorldCup2018/frames/6e577252c4004961ac7caa738a52c238'

    # feature
    feature_path = imgs_path.replace("frames", "features") + '.pkl'
    video_features = pickle.load(open(feature_path, 'rb'))

    # proposal
    basename = imgs_path.replace('frames', 'mp4') + '.mp4'
    bmn_results = prop_dict[basename]

    material = {'feature': video_features, 'proposal': bmn_results}

    t0 = time.time()
    outputs = model.predict(cfg, material)
    # outputs = model.infer(np.random.rand(32, 8, 3, 224, 224).astype(np.float32))
    # print(outputs.shape)
    t1 = time.time()
    results = {'actions': outputs}
    with open('results.json', 'w', encoding='utf-8') as f:
       data = json.dumps(results, indent=4, ensure_ascii=False)
       f.write(data) 

    print('cost time = {} min'.format((t1 - t0) / 60.0))
