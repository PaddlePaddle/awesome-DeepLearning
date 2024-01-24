import os
import json
import time
import sys
import paddle

from data import COCODataSet, BaseDataLoader
from data.operators import *
from models import ComposeCallback, LogPrinter

def get_categories(metric_type, anno_file=None, arch=None):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        anno_file (str): annotation file path
    """
    if anno_file and os.path.isfile(anno_file):
        # lazy import pycocotools here
        from pycocotools.coco import COCO

        coco = COCO(anno_file)
        cats = coco.loadCats(coco.getCatIds())

        clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
        catid2name = {cat['id']: cat['name'] for cat in cats}
        return clsid2catid, catid2name

    # anno file not exist, load default categories of COCO17
    else:
        return _coco17_category()

def _coco17_category():
    """
    Get class id to category id map and category id
    to category name map of COCO2017 dataset

    """
    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    catid2name = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }

    clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
    catid2name.pop(0)

    return clsid2catid, catid2name

def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    return infer_res
def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """
    assert coco_gt != None or anno_file != None

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if coco_gt == None:
        coco_gt = COCO(anno_file)
    print("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)

    coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats

class COCOMetric(paddle.metric.Metric):
    def __init__(self, anno_file, **kwargs):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories('COCO', anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.iou_type = kwargs.get('IouType', 'bbox')
        self.reset()
    
    def name(self):
        return self.__class__.__name__

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []

    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                print('The bbox result is saved to bbox.json.')

            bbox_stats = cocoapi_eval(
                output,
                'bbox',
                anno_file=self.anno_file,
                classwise=self.classwise)
            self.eval_results['bbox'] = bbox_stats
            sys.stdout.flush()

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:
    def log(self):
        pass

    def get_results(self):
        return self.eval_results

def _init_metrics(dataset):
    # pass clsid2catid info to metric instance to avoid multiple loading
    # annotation file
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()} 

    # when do validation in train, annotation file should be get from
    # EvalReader instead of self.dataset(which is TrainReader)
    anno_file = dataset.get_anno()

    _metrics = [
        COCOMetric(
            anno_file=anno_file,
            clsid2catid=clsid2catid,
            classwise=False,
            output_eval=None,
            bias=0,
            IouType='bbox',
            save_prediction_only=False)
    ]
    return _metrics

def _reset_metrics(_metrics):
    for metric in _metrics:
        metric.reset()


def _eval_with_loader(model,dataset_dir,image_dir,anno_path):
    status = {}
    _callbacks = [LogPrinter(model)]
    _compose_callback = ComposeCallback(_callbacks)    

    dataset = COCODataSet(dataset_dir=dataset_dir, image_dir=image_dir,anno_path=anno_path)
    _eval_batch_sampler = paddle.io.BatchSampler(dataset, batch_size=1)
    
    sample_transforms = [{Decode: {}}, {Resize: {'target_size': [800, 1333], 'keep_ratio': True}}, {NormalizeImage: {'is_scale': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, {Permute: {}}]
    batch_transforms = [{PadMaskBatch:{'pad_to_stride': -1, 'return_pad_mask': True}}]
    loader = BaseDataLoader(sample_transforms, batch_transforms, batch_size=1, shuffle=False, drop_last=False, drop_empty=False)(dataset, 4, _eval_batch_sampler)


    _metrics = _init_metrics(dataset=dataset)

    sample_num = 0
    tic = time.time()
    _compose_callback.on_epoch_begin(status)
    status['mode'] = 'eval'
    model.eval()
    for step_id, data in enumerate(loader):
        status['step_id'] = step_id
        _compose_callback.on_step_begin(status)
        # forward
        outs = model(data)

        # update metrics
        for metric in _metrics:
            metric.update(data, outs)

        sample_num += data['im_id'].numpy().shape[0]
        _compose_callback.on_step_end(status)

    status['sample_num'] = sample_num
    status['cost_time'] = time.time() - tic

    # accumulate metric to log out
    for metric in _metrics:
        metric.accumulate()
        metric.log()
    _compose_callback.on_epoch_end(status)
    # reset metric states for metric may performed multiple times
    _reset_metrics(_metrics)

def evaluate(model,dataset_dir,image_dir,anno_path):
    with paddle.no_grad():
        _eval_with_loader(model,dataset_dir,image_dir,anno_path)
