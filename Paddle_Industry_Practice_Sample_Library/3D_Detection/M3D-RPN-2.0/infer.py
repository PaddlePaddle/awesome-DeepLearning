import paddle
from lib.util import *
from lib.rpn_util import *
import argparse
from data.augmentations import *
from easydict import EasyDict as edict
from data.m3drpn_reader import read_kitti_cal
from paddle.fluid.dygraph import to_variable
from lib.nms.gpu_nms import gpu_nms
import pdb

def parse_args():
    """parse"""
    parser = argparse.ArgumentParser("M3D-RPN train script")
    parser.add_argument("--conf_path", type=str, default='', help="config.pkl")
    parser.add_argument(
        '--weights_path', type=str, default='', help='weights save path')

    parser.add_argument(
        '--backbone',
        type=str,
        default='DenseNet121',
        help='backbone model to train, default DenseNet121')

    parser.add_argument(
        '--data_dir', type=str, default='dataset', help='dataset directory')

    args = parser.parse_args()
    return args

def decode_infer(cls, prob, bbox_2d, bbox_3d, feat_size, rois, rpn_conf, synced=False):
    gpu = 0
    # compute feature resolution
    num_anchors = rpn_conf.anchors.shape[0]

    bbox_x = bbox_2d[:, :, 0]
    bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]
    bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]
    bbox_y3d = bbox_3d[:, :, 1]
    bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]
    bbox_h3d = bbox_3d[:, :, 4]
    bbox_l3d = bbox_3d[:, :, 5]
    bbox_ry3d = bbox_3d[:, :, 6]

    # detransform 3d
    bbox_x3d = bbox_x3d * rpn_conf.bbox_stds[:, 4][
        0] + rpn_conf.bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * rpn_conf.bbox_stds[:, 5][
        0] + rpn_conf.bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * rpn_conf.bbox_stds[:, 6][
        0] + rpn_conf.bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * rpn_conf.bbox_stds[:, 7][
        0] + rpn_conf.bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * rpn_conf.bbox_stds[:, 8][
        0] + rpn_conf.bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * rpn_conf.bbox_stds[:, 9][
        0] + rpn_conf.bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * rpn_conf.bbox_stds[:, 10][
        0] + rpn_conf.bbox_means[:, 10][0]

    # find 3d source

    #tracker = rois[:, 4].cpu().detach().numpy().astype(np.int64)
    #src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).cuda().type(torch.cuda.FloatTensor)
    tracker = rois[:, 4].astype(np.int64)
    src_3d = rpn_conf.anchors[tracker, 4:]

    #tracker_sca = rois_sca[:, 4].cpu().detach().numpy().astype(np.int64)
    #src_3d_sca = torch.from_numpy(rpn_conf.anchors[tracker_sca, 4:]).cuda().type(torch.cuda.FloatTensor)

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d_np = bbox_x3d
    bbox_y3d_np = bbox_y3d  #(1, N)
    bbox_z3d_np = bbox_z3d
    bbox_w3d_np = bbox_w3d
    bbox_l3d_np = bbox_l3d
    bbox_h3d_np = bbox_h3d
    bbox_ry3d_np = bbox_ry3d

    bbox_x3d_np = bbox_x3d_np[0, :] * widths + ctr_x
    bbox_y3d_np = bbox_y3d_np[0, :] * heights + ctr_y

    bbox_x_np = bbox_x
    bbox_y_np = bbox_y
    bbox_w_np = bbox_w
    bbox_h_np = bbox_h

    bbox_z3d_np = src_3d[:, 0] + bbox_z3d_np[0, :]  #(N, 5), (N2, 1)
    bbox_w3d_np = np.exp(bbox_w3d_np[0, :]) * src_3d[:, 1]
    bbox_h3d_np = np.exp(bbox_h3d_np[0, :]) * src_3d[:, 2]
    bbox_l3d_np = np.exp(bbox_l3d_np[0, :]) * src_3d[:, 3]
    bbox_ry3d_np = src_3d[:, 4] + bbox_ry3d_np[0, :]

    # bundle
    coords_3d = np.stack((bbox_x3d_np, bbox_y3d_np, bbox_z3d_np[:bbox_x3d_np.shape[0]], bbox_w3d_np[:bbox_x3d_np.shape[0]], bbox_h3d_np[:bbox_x3d_np.shape[0]], \
        bbox_l3d_np[:bbox_x3d_np.shape[0]], bbox_ry3d_np[:bbox_x3d_np.shape[0]]), axis=1)#[N, 7]

    # compile deltas pred

    deltas_2d = np.concatenate(
        (bbox_x_np[0, :, np.newaxis], bbox_y_np[0, :, np.newaxis],
         bbox_w_np[0, :, np.newaxis], bbox_h_np[0, :, np.newaxis]),
        axis=1)  #N,4
    coords_2d = bbox_transform_inv(
        rois,
        deltas_2d,
        means=rpn_conf.bbox_means[0, :],
        stds=rpn_conf.bbox_stds[0, :])  #[N,4]

    # detach onto cpu
    #coords_2d = coords_2d.cpu().detach().numpy()
    #coords_3d = coords_3d.cpu().detach().numpy()
    prob_np = prob[0, :, :]  #.cpu().detach().numpy()

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    cls_pred = np.argmax(prob_np[:, 1:], axis=1) + 1
    scores = np.amax(prob_np[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    if synced:

        # nms
        keep_inds = gpu_nms(
            aboxes[:, 0:5].astype(np.float32),
            rpn_conf.nms_thres,
            device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    else:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        tracker = tracker[0:min(rpn_conf.nms_topN_pre, tracker.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
        coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

        # nms
        keep_inds = gpu_nms(
            aboxes[:, 0:5].astype(np.float32),
            rpn_conf.nms_thres,
            device_id=gpu)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d,
                            tracker[:, np.newaxis]))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)
    return aboxes

if __name__ == "__main__":
    ### config ###
    args = parse_args()
    conf = edict(pickle_read(args.conf_path))
    results_path = os.path.join('./inference_result')
    mkdir_if_missing(results_path, delete_if_exist=True)
    imlist = list_files(
        os.path.join(args.data_dir, conf.dataset_test, 'validation', 'image_2', ''),
        '*.png')
    preprocess = Preprocess([conf.test_scale,1760], conf.image_means,
                            conf.image_stds)

    ### model prepare ###
    paddle.enable_static()
    path_prefix = "./inference/model"
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    [inference_program, feed_target_name, fetch_targets] = (paddle.static.load_inference_model(path_prefix,exe))

    for imind, impath in enumerate(imlist):
        im = cv2.imread(impath)
        base_path, name, ext = file_parts(impath)
        # read in calib
        p2 = read_kitti_cal(
            os.path.join(args.data_dir, conf.dataset_test, 'validation', 'calib', name +'.txt'))
        p2_inv = np.linalg.inv(p2)

        ## preprocess ##
        imH_orig = im.shape[0]
        imW_orig = im.shape[1]
        im = preprocess(im)
        im = im[np.newaxis, :, :, :]
        imH = im.shape[2]
        imW = im.shape[3]
        # im = paddle.to_tensor(im)
        scale_factor = imH / imH_orig
        cls, prob, bbox_2d, bbox_3d, feat_size = exe.run(inference_program, feed={feed_target_name[0]:im},fetch_list=fetch_targets)
        tmp_feat_size = calc_output_size(
            np.array(conf.crop_size), conf.feat_stride)
        rois = locate_anchors(conf.anchors, tmp_feat_size, conf.feat_stride)
        aboxes = decode_infer(cls, prob, bbox_2d, bbox_3d, feat_size, rois, conf)

        base_path, name, ext = file_parts(impath)
        file = open(os.path.join(results_path, name + '.txt'), 'w')
        text_to_write = ''
        for boxind in range(0, min(conf.nms_topN_post, aboxes.shape[0])):
            box = aboxes[boxind, :]
            score = box[4]
            cls = conf.lbls[int(box[5] - 1)]
            if score >= 0.5: 

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                # convert alpha into ry3d
                coord3d = np.linalg.inv(p2).dot(
                    np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

                step_r = 0.3 * math.pi
                r_lim = 0.01
                box_2d = np.array([x1, y1, width, height])

                z3d, ry3d, verts_best = hill_climb(
                    p2,
                    p2_inv,
                    box_2d,
                    x3d,
                    y3d,
                    z3d,
                    w3d,
                    h3d,
                    l3d,
                    ry3d,
                    step_r_init=step_r,
                    r_lim=r_lim)

                # predict a more accurate projection
                coord3d = np.linalg.inv(p2).dot(
                    np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])

                x3d = coord3d[0]
                y3d = coord3d[1]
                z3d = coord3d[2]

                y3d += h3d / 2

                text_to_write += (
                    '{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                    + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d,
                                                w3d, l3d, x3d, y3d, z3d, ry3d,
                                                score)

        file.write(text_to_write)
        file.close()
        # pdb.set_trace()
