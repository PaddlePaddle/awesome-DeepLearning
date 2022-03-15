# 足球动作检测模型


## 内容
- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推理](#模型推理)
- [模型优化](#模型优化)
- [模型部署](#模型部署)
- [参考论文](#参考论文)


## 模型简介
该代码库用于体育动作检测+识别, 基于paddle2.0版本开发，结合PaddleVideo中的ppTSM, BMN, attentionLSTM的多个视频模型进行视频时空二阶段检测算法。
主要分为如下几步
 - 特征抽取
    - 图像特性，ppTSM
    - 音频特征，Vggsound
 - proposal提取，BMN
 - LSTM，动作分类 + 回归


## 数据准备
- 数据集label格式
```
数据集来自欧洲杯2016，共49个足球视频，其中训练集44个，验证集5个
数据集地址: datasets/EuroCup2016/dataset_url.list
数据集label格式
{
    "0": "背景",
    "1": "进球",
    "2": "角球",
    "3": "任意球",
    "4": "黄牌",
    "5": "红牌",
    "6": "换人",
    "7": "界外球",
}
数据集标注文件:
datasets/EuroCup2016/label_cls8_train.json
datasets/EuroCup2016/label_cls8_val.json
```

- 数据集gts处理, 将原始标注数据处理成如下json格式
```
{
    'fps': 5,
    'gts': [
        {
            'url': 'xxx.mp4',
            'total_frames': 6341,
            'actions': [
                {
                    "label_ids": [7],
                    "label_names": ["界外球"],
                    "start_id": 395,
                    "end_id": 399
                },
                ...
            ]
        },
        ...
    ]
}
```

- 数据集抽帧, 由mp4, 得到frames和pcm, 这里需要添加ffmpeg环境
```
cd datasets/script && python get_frames_pcm.py
```
- 数据预处理后保存格式如下
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # xx数据集
            |--  mp4               # 原始视频.mp4
            |--  frames            # 图像帧, fps=5, '.jpg'格式
            |--  pcm               # 音频pcm, 音频采样率16000，采用通道数1
            |--  url.list          # 视频列表
            |--  label_train.json  # 训练集原始gts
            |--  label_val.json    # 验证集原始gts
```


## 模型训练
采样方式：
- image 采样频率fps=5，如果有些动作时间较短，可以适当提高采样频率
- BMN windows=200，即40s，所以测试自己的数据时，视频时长需大于40s

### 基础镜像
```
docker pull tmtalgo/paddleaction:action-detection-v2
```

### 代码结构
```
|-- root_dir
   |--  checkpoints                # 保存训练后的模型和log
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016           # xx数据集
            |--  feature_bmn       # bmn提取到的proposal
            |--  features          # tsn和audio特征, image fps=5, audio 每秒(1024)
            |--  input_for_bmn     # bmn训练的输入数据，widows=40
            |--  input_for_lstm    # lstm训练的输入数据
            |--  input_for_tsn     # tsn训练的数据数据
            |--  mp4               # 原始视频.mp4
            |--  frames            # 图像帧, fps=5, '.jpg'格式
            |--  pcm               # 音频pcm, 音频采样率16000，采用通道数1
            |--  url.list          # 视频列表
            |--  label_train.json  # 训练集原始gts
            |--  label_val.json    # 验证集原始gts
        |--  script                # 数据集处理脚本
    |--  predict                   # 模型预测代码
    |--  extractor                 # 特征提取脚本
    |--  train_lstm                # lstm训练代码
    |--  train_proposal            # tsn、bmn训练代码，基本保持paddle-release-v1.8版本，数据接口部分稍有改动，参考官网
        |--  configs               # tsn and bmn football config file
    |--  train_tsn.sh              # tsn训练启动脚本
    |--  train_bmn.sh              # bmn训练启动脚本
    |--  train_lstm.sh             # lstm训练启动脚本
```

### step1 ppTSM训练

#### step1.1  ppTSM 训练数据处理
由frames结合gts生成训练所需要的正负样本
```
cd datasets/script && python get_instance_for_tsn.py

# 文件名按照如下格式
'{}_{}_{}_{}'.format(video_basename, start_id, end_id, label)
```
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016           # xx数据集
            |--  input_for_tsn     # tsn/tsm训练的数据
```

#### step1.2 ppTSM模型训练
我们提供了足球数据训练的模型，参考checkpoints
如果需要在自己的数据上训练，可参考
https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
config.yaml参考configs文件夹下pptsm_football_v2.0.yaml
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
cd ${PaddleVideo}
python -B -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    --log_dir=$save_dir/logs \
    main.py  \
    --validate \
    -c ${FootballAcation}/train_proposal/configs/pptsm_football_v2.0.yaml \
    -o output_dir=$save_dir
```

#### step1.3 ppTSM模型转为预测模式
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
cd ${PaddleVideo}
python tools/export_model.py -c ${FootballAcation}/train_proposal/configs/pptsm_football_v2.0.yaml \
                               -p ${pptsm_train_dir}/checkpoints/models_pptsm/ppTSM_epoch_00057.pdparams \
                               -o {FootballAcation}/checkpoints/ppTSM
```

####  step1.4  基于ppTSM的视频特征提取
image and audio特征提取，保存到datasets features文件夹下
```
cd ${FootballAcation}
cd extractor && python extract_feat.py
# 特征维度, image(2048) + audio(1024)
# 特征保存格式如下，将如下dict保存在pkl格式，用于接下来的BMN训练
video_features = {'image_feature': np_image_features,
                  'audio_feature': np_audio_features}
```
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # xx数据集
            |--  features          # 视频的图像+音频特征
```


### step2 BMN训练
BMN训练代码为：https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
BMN文档参考：https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/localization/bmn.md

#### step2.1 BMN训练数据处理
用于提取二分类的proposal，windows=40，根据gts和特征得到BMN训练所需要的数据集
```
cd datasets/script && python get_instance_for_bmn.py
# 数据格式
{
    "719b0a4bcb1f461eabb152298406b861_753_793": {
        "duration_second": 40.0,
        "duration_frame": 200,
        "feature_frame": 200,
        "subset": "train",
        "annotations": [
            {
                "segment": [
                    15.0,
                    22.0
                ],
                "label": "3.0",
                "label_name": "任意球"
            }
        ]
    },
    ...
}
```
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # xx数据集
            |--  input_for_bmn     # bmn训练的proposal         
```

#### step2.2  BMN模型训练
我们同样提供了足球数据训练的模型，参考checkpoints
如果要在自己的数据上训练，步骤与step1.2 ppTSM 训练相似，可参考
https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
config.yaml参考configs文件夹下bmn_football_v2.0.yaml
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
cd ${PaddleVideo}
python -B -m paddle.distributed.launch \
     --gpus="0,1" \
     --log_dir=$out_dir/logs \
     main.py  \
     --validate \
     -c ${FootballAcation}/train_proposal/configs/bmn_football_v2.0.yaml \
     -o output_dir=$out_dir
```

#### step2.3 BMN模型转为预测模式
同step1.3
```
# https://github.com/PaddlePaddle/PaddleVideo/tree/release/2.0
$cd {PaddleVideo}
python tools/export_model.py -c ${FootballAcation}/train_proposal/configs/bmn_football_v2.yaml \
                               -p ${bmn_train_dir}/checkpoints/models_bmn/bmn_epoch16.pdparams \
                               -o {FootballAcation}/checkpoints/ppTSM
```

#### step2.4  BMN模型预测
得到动作proposal信息： start_id, end_id, score
```
cd extractor && python extract_bmn.py
# 数据格式
[
    {
        "video_name": "c9516c903de3416c97dae91a59e968d7",
        "num_proposal": 5534,
        "bmn_results": [
            {
                "start": 7850.0,
                "end": 7873.0,
                "score": 0.77194699622342
            },
            {
                "start": 4400.0,
                "end": 4443.0,
                "score": 0.7663803287641536
            },
            ...
        ]
    },
    ...
]
```
完成该步骤后，数据存储位置
```
   |--  datasets                   # 训练数据集和处理脚本
        |--  EuroCup2016            # xx数据集
            |--  feature_bmn
                 |--  prop.json    # bmn 预测结果
```

### step3 LSTM训练

#### step3.1  LSTM训练数据处理
将BMN得到的proposal截断并处理成LSTM训练所需数据集
```
cd datasets/script && python get_instance_for_lstm.py
# 数据格式1，label_info
{
    "fps": 5,
    "results": [
        {
            "url": "https://xxx.mp4",
            "mode": "train",        # train or validation
            "total_frames": 6128,
            "num_gts": 93,
            "num_proposals": 5043,
            "proposal_actions": [
                {
                    "label": 6,
                    "norm_iou": 0.7575757575757576,
                    "norm_ioa": 0.7575757575757576,
                    "norm_start": -0.32,
                    "proposal": {
                        "start": 5011,
                        "end": 5036,
                        "score": 0.7723643666324231
                    },
                    "hit_gts": {
                        "label_ids": [
                            6
                        ],
                        "label_names": [
                            "换人"
                        ],
                        "start_id": 5003,
                        "end_id": 5036
                    }
                },
                ...
        },
        ...
}
# 数据格式2，LSTM训练所需要的feature
{
    'features': np.array(feature_hit, dtype=np.float32),    # TSN and audio 特征
    'feature_fps': 5,                                       # fps = 5
    'label_info': {'norm_iou': 0.5, 'label': 3, ...},       # 数据格式1中的'proposal_actions'
    'video_name': 'c9516c903de3416c97dae91a59e968d7'        # video_name
}
# 数据格式3，LSTM训练所需label.txt
'{} {}'.format(filename, label)
```

#### step3.2  LSTM训练
```
sh run.sh	# LSTM 模块
```

#### step3.3 LSTM模型转为预测模式
```
${FootballAction}
python tools/export_model.py -c ${FootballAction}/train_lstm/conf/conf.yaml \
                               -p ${lstm_train_dir}/checkpoints/models_lstm/bmn_epoch29.pdparams \
                               -o {FootballAction}/checkpoints/LSTM
```

## 模型推理
运行预测代码
```
cd predict && python predict.py
```
产出文件：results.json


## 模型评估
```
# 包括bmn proposal 评估和最终action评估
cd predict && python eval.py results.json
```


## 模型优化
- 基础特征模型（图像）替换为ppTSM，准确率由84%提升到94%
- 基础特征模型（音频）没变动
- BMN，请使用paddlevideo最新版
- LSTM，暂时提供v1.8训练代码（后续升级为v2.0），也可自行尝试使用paddlevideo-2.0中的attentation lstm
- 为兼容paddle-v1.8和paddle-v2.0，将模型预测改为inference model，训练代码可以使用v1.8或v2.0，只要export为inference model即可进行预测
- 准确率提升，precision和recall均有大幅提升，F1-score从0.57提升到0.82


## 模型部署
本代码解决方案在动作的检测和召回指标F1-score=82%


## 参考论文
- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/abs/1609.08675), Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan