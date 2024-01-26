import torch
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device
import random
from utils.plots import plot_one_box
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class Detector(): 
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()

    def init_model(self):
        self.weights = 'weights/dagm_s.pt'
        
        # Initialize
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            model.half()  # to FP16
        self.model = model

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.conf_thres = 0.5
        self.img_size = 640
        self.iou_thres = 0.5
        
    def preprocess(self, img):
        # Padded resize
        img = letterbox(img, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect(self, im):
        img = self.preprocess(im)
        
        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        #pred = pred.float()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        pred_boxes = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im.shape).round()

                for *xyxy, conf, cls_id in reversed(det):
                    # Add bbox to image
                    cls = self.names[int(cls_id)]
                    label = '%s %.2f' % (cls, conf)
                    plot_one_box(xyxy, im, label=label, color=self.colors[int(cls_id)], line_thickness=3)

                    pred_boxes.append((xyxy, conf, cls))
    
        return im, pred_boxes

