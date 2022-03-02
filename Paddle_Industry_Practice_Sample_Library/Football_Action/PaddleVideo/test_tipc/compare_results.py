import numpy as np
import os
import subprocess
import json
import argparse
import glob


def init_args():
    parser = argparse.ArgumentParser()
    # params for testing assert allclose
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--gt_file", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--precision", type=str, default="fp32")
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def run_shell_command(cmd):
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True)
    out, err = p.communicate()

    if p.returncode == 0:
        return out.decode('utf-8')
    else:
        return None


def parser_results_from_log_by_name(log_path, names_list):
    if not os.path.exists(log_path):
        raise ValueError("The log file {} does not exists!".format(log_path))

    if names_list is None or len(names_list) < 1:
        return []

    parser_results = {}
    lines = open(log_path, 'r').read().splitlines()
    if 'python_infer' in log_path:  # parse python inference
        for line in lines:
            split_items = line.replace('\t', ' ')
            split_items = split_items.split(' ')
            split_items = [item for item in split_items if len(item) > 0]
            for name in names_list:
                if name in line:
                    if '.' in split_items[-1]:
                        parser_results[name] = float(split_items[-1])
                    else:
                        parser_results[name] = int(split_items[-1])
    else:  # parse cpp inference
        for line in lines:
            split_items = line.replace('\t', ' ')
            split_items = split_items.split(' ')
            split_items = [item for item in split_items if len(item) > 0]
            if all([(name + ':') in split_items for name in names_list]):
                # print(split_items)
                parser_results['class'] = int(split_items[2])
                parser_results['score'] = float(split_items[-1])
    return parser_results


def load_gt_from_file(gt_file):
    if not os.path.exists(gt_file):
        raise ValueError("The log file {} does not exists!".format(gt_file))
    with open(gt_file, 'r') as f:
        data = f.readlines()
        f.close()
    parser_gt = {}
    for line in data:
        if 'top-1 class' in line:
            split_items = line.replace('\t', ' ')
            split_items = split_items.split(' ')
            split_items = [item for item in split_items if len(item) > 0]
            parser_gt['top-1 class'] = int(split_items[-1])
        elif 'top-1 score' in line:
            split_items = line.replace('\t', ' ')
            split_items = split_items.split(' ')
            split_items = [item for item in split_items if len(item) > 0]
            parser_gt['top-1 score'] = float(split_items[-1])
        elif "score" in line and 'segment' in line:
            location_dict = eval(line)
            parser_gt[f"score_{len(parser_gt)}"] = location_dict['score']
            parser_gt[f"segment_{len(parser_gt)}"] = location_dict['segment']
        elif "class:" in line and "score:" in line:
            split_items = line.replace('\t', ' ')
            split_items = split_items.split(' ')
            split_items = [item for item in split_items if len(item) > 0]
            parser_gt['class'] = int(split_items[2])
            parser_gt['score'] = float(split_items[-1])
    return parser_gt


def load_gt_from_txts(gt_file):
    gt_list = glob.glob(gt_file)
    gt_collection = {}
    for gt_f in gt_list:
        gt_dict = load_gt_from_file(gt_f)
        basename = os.path.basename(gt_f)
        if "fp32" in basename:
            gt_collection["fp32"] = [gt_dict, gt_f]
        elif "fp16" in basename:
            gt_collection["fp16"] = [gt_dict, gt_f]
        elif "int8" in basename:
            gt_collection["int8"] = [gt_dict, gt_f]
        else:
            continue
    return gt_collection


def collect_predict_from_logs(log_path, key_list):
    log_list = glob.glob(log_path)
    pred_collection = {}
    for log_f in log_list:
        pred_dict = parser_results_from_log_by_name(log_f, key_list)
        key = os.path.basename(log_f)
        pred_collection[key] = pred_dict

    return pred_collection


def testing_assert_allclose(dict_x, dict_y, atol=1e-7, rtol=1e-7):
    for k in dict_x:
        np.testing.assert_allclose(np.array(dict_x[k]),
                                   np.array(dict_y[k]),
                                   atol=atol,
                                   rtol=rtol)


if __name__ == "__main__":
    # Usage example:
    # test python infer:
    ## python3.7 test_tipc/compare_results.py --gt_file=./test_tipc/results/PP-TSM/*.txt  --log_file=./test_tipc/output/PP-TSM/python_infer_*.log
    # test cpp infer:
    ## python3.7 test_tipc/compare_results.py --gt_file=./test_tipc/results/PP-TSM_CPP/*.txt  --log_file=./test_tipc/output/PP-TSM_CPP/cpp_infer_*.log

    args = parse_args()

    gt_collection = load_gt_from_txts(args.gt_file)
    key_list = gt_collection["fp32"][0].keys()
    pred_collection = collect_predict_from_logs(args.log_file, key_list)
    for filename in pred_collection.keys():
        if "fp32" in filename:
            gt_dict, gt_filename = gt_collection["fp32"]
        elif "fp16" in filename:
            gt_dict, gt_filename = gt_collection["fp16"]
        elif "int8" in filename:
            gt_dict, gt_filename = gt_collection["int8"]
        else:
            continue
        pred_dict = pred_collection[filename]
        try:
            testing_assert_allclose(gt_dict,
                                    pred_dict,
                                    atol=args.atol,
                                    rtol=args.rtol)
            print(
                "Assert allclose passed! The results of {} and {} are consistent!"
                .format(filename, gt_filename))
        except Exception as E:
            print(E)
            raise ValueError(
                "The results of {} and the results of {} are inconsistent!".
                format(filename, gt_filename))
