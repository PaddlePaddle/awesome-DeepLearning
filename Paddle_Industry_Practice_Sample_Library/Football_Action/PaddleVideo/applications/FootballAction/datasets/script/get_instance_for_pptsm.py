"""
get instance for pptsm
positive: 标注后的动作区间，一个区间所有frames生成一个pkl
negative: 标注后的非动作区间，随机取N个区间生成N个pkl，每个区间长度等于最近的前一个动作区间的长度
"""
import os
import json
import numpy as np
import random
import pickle
from concurrent import futures

dataset = "/home/PaddleVideo/applications/FootballAction/datasets/EuroCup2016"
frames_dir = dataset + '/frames'
label_files = {'train': 'label_cls8_train.json', 'val': 'label_cls8_val.json'}


def process(item, fps, save_folder):
    actions_pos = []
    actions_neg = []
    url = item['url']
    print(url)
    basename = os.path.basename(url).split('.')[0]
    actions = item['actions']
    # pos
    for action in actions:
        actions_pos.append({
            'label': action['label_ids'],
            'start': action['start_id'] * fps,
            'end': action['end_id'] * fps
        })
    # neg
    for idx, pos in enumerate(actions_pos):
        if idx == len(actions_pos) - 1:
            break
        len_pos = pos['end'] - pos['start']
        duration_start = [pos['end'], actions_pos[idx + 1]['start'] - len_pos]
        if duration_start[1] - duration_start[0] < 3:
            continue
        for k in range(1, 3):
            start_frame = random.randint(duration_start[0], duration_start[1])
            end_frame = start_frame + len_pos
            actions_neg.append({
                'label': [0],
                'start': start_frame,
                'end': end_frame
            })
    # save pkl
    for item in np.concatenate((actions_pos, actions_neg), axis=0):
        start = item['start']
        end = item['end']
        label = item['label']
        label_str = str(label[0])
        if len(item['label']) == 2:
            label_str = label_str + '-' + str(label[1])
        frames = []
        for ii in range(start, end + 1):
            img = os.path.join(frames_dir, basename, '%08d.jpg' % ii)
            with open(img, 'rb') as f:
                data = f.read()
            frames.append(data)
        # print(label_str)
        outname = '%s/%s_%08d_%08d_%s.pkl' % (save_folder, basename, start, end,
                                              label_str)
        with open(outname, 'wb') as f:
            pickle.dump((basename, label, frames), f, -1)


def gen_instance_pkl(label_data, save_folder):
    fps = label_data['fps']
    gts = label_data['gts']
    with futures.ProcessPoolExecutor(max_workers=10) as executer:
        fs = [executer.submit(process, gt, fps, save_folder) for gt in gts]

    #for gt in gts:
    #    process(gt, fps, save_folder)


if __name__ == "__main__":
    for item, value in label_files.items():
        save_folder = os.path.join(dataset, 'input_for_pptsm', item)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        label_file = os.path.join(dataset, value)
        label_data = json.load(open(label_file, 'rb'))

        gen_instance_pkl(label_data, save_folder)

    # gen train val list
    data_dir = '/home/PaddleVideo/applications/FootballAction/datasets/EuroCup2016/input_for_pptsm/'
    os.system('find ' + data_dir + 'train -name "*.pkl" > ' + data_dir +
              'train.list')
    os.system('find ' + data_dir + 'val -name "*.pkl" > ' + data_dir +
              'val.list')
