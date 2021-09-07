import argparse
from models import ResNet, DETRTransformer, HungarianMatcher, DETRLoss, DETRHead, DETRBBoxPostProcess,DETR

from utils import load_weights
from train_model import train
from eval_model import evaluate
from test_model import get_test_images, predict
# build model. 
def build_model():
    backbone = ResNet(depth=50, norm_type='bn', freeze_at=0, return_idx=[3], lr_mult_list=[0.0, 0.1, 0.1, 0.1], num_stages=4)

    transformer = DETRTransformer(num_queries=100, position_embed_type='sine', nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', hidden_dim=256, backbone_num_channels=2048)

    matcher = HungarianMatcher(matcher_coeff={'class': 1, 'bbox': 5, 'giou': 2}, use_focal_loss=False)

    loss = DETRLoss(loss_coeff={'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}, aux_loss=True, num_classes=80, use_focal_loss=False, matcher=matcher)

    detr_head = DETRHead(num_mlp_layers=3, num_classes=80, hidden_dim=256, use_focal_loss=False, nhead=8, fpn_dims=[], loss=loss)

    post_process = DETRBBoxPostProcess(num_classes=80, use_focal_loss=False)

    model = DETR(backbone=backbone,
                    transformer=transformer,
                    detr_head=detr_head,
                    post_process=post_process)
    return model
# python main.py --mode='train' --dataset_dir='dataset/' --image_dir='train2017' --anno_path='annotations/instances_train2017.json'
# python main.py --mode='eval' --dataset_dir='dataset/' --image_dir='val2017' --anno_path='annotations/instances_val2017.json'
# python main.py --mode=='test' --infer_img='test_imgs/000000014439.jpg'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str, default='train', help='choose mode for train or eval or test')
    parser.add_argument('--dataset_dir',type=str, default='dataset/', help='dir of datset')
    parser.add_argument('--image_dir',type=str, default='train2017/', help='dir of datset images')
    parser.add_argument('--anno_path',type=str, default='annotions/instances_train2017', help='json file')
    parser.add_argument('--infer_img',type=str, default='test_imgs/000000014439.jpg', help='test image')
    parser.add_argument('--pretrained_model',type=str,default='pretrained_model/detr',help='pretrained model')
    
    args = parser.parse_args()

    model = build_model()
    # 模型训练
    if args.mode == 'train':
        start_epoch = 0
        end_epoch = 500
        train(model, start_epoch, end_epoch,
              dataset_dir=args.dataset_dir,
              image_dir=args.image_dir,
              anno_path=args.anno_path)

    if args.mode == 'eval':
        # 模型评估
        # load weights,predict,eval
        load_weights(model, args.pretrained_model)
        evaluate(model,
                 dataset_dir=args.dataset_dir,
                 image_dir=args.image_dir,
                 anno_path=args.anno_path)

    if args.mode == 'test':
        # 模型测试
        # load weights,predict,eval
        load_weights(model, args.pretrained_model)

        images = get_test_images(args.infer_img)
        predict(
            images,
            model,
            draw_threshold=0.5,
            output_dir="output",
            anno_path=args.anno_path)