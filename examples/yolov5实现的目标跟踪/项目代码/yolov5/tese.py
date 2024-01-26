import track as tk
import torch

parser = tk.argparse.ArgumentParser()
parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true',
                        help='display results')
parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
args = parser.parse_args()
args.img_size = tk.check_img_size(args.img_size)
print(args)

with torch.no_grad():
    tk.detect(args)