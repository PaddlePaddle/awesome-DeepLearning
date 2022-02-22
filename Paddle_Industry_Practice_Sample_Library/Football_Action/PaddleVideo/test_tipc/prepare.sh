#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# set -xe

:<<!
MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer',
#                 'cpp_infer', ]
!

MODE=$2

dataline=$(cat ${FILENAME})

python3.7 -m pip install unrar

git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog
python3.7 -m pip install -r requirements.txt
python3.7 setup.py bdist_wheel
python3.7 -m pip install ./dist/auto_log-1.0.0-py3-none-any.whl
cd ..

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")




if [ ${MODE} = "lite_train_lite_infer" ];then
    if [ ${model_name} == "PP-TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "PP-TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "STGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        pushd data/yt8m
        ## download & decompression training data
        wget -nc https://videotag.bj.bcebos.com/Data/yt8m_rawframe_small.tar
        tar -xf yt8m_rawframe_small.tar
        python3.7 -m pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
        python3.7 tf2pkl.py ./frame ./pkl_frame/
        ls pkl_frame/train*.pkl > train_small.list # 将train*.pkl的路径写入train_small.list
        ls pkl_frame/validate*.pkl > val_small.list # 将validate*.pkl的路径写入val_small.list

        python3.7 split_yt8m.py train_small.list # 拆分每个train*.pkl变成多个train*_split*.pkl
        python3.7 split_yt8m.py val_small.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl

        ls pkl_frame/train*_split*.pkl > train_small.list # 将train*_split*.pkl的路径重新写入train_small.list
        ls pkl_frame/validate*_split*.pkl > val_small.list # 将validate*_split*.pkl的路径重新写入val_small.list
        popd
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain lite train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    else
        echo "Not added into TIPC yet."
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    if [ ${model_name} == "PP-TSM" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        python3.7 extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4 # extract frames from video file
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "PP-TSN" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # pretrain whole train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "STGCN" ]; then
        # pretrain whole train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "TSM" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        python3.7 extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4 # extract frames from video file
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        python3.7 extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext mp4 # extract frames from video file
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        # pretrain whole train data
        pushd data/yt8m
        mkdir frame
        cd frame
        ## download & decompression training data
        curl data.yt8m.org/download.py | partition=2/frame/train mirror=asia python
        curl data.yt8m.org/download.py | partition=2/frame/validate mirror=asia python
        python3.7 -m pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
        cd ..
        python3.7 tf2pkl.py ./frame ./pkl_frame/
        ls pkl_frame/train*.pkl > train.list # 将train*.pkl的路径写入train.list
        ls pkl_frame/validate*.pkl > val.list # 将validate*.pkl的路径写入val.list

        python3.7 split_yt8m.py train.list # 拆分每个train*.pkl变成多个train*_split*.pkl
        python3.7 split_yt8m.py val.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl

        ls pkl_frame/train*_split*.pkl > train.list # 将train*_split*.pkl的路径重新写入train.list
        ls pkl_frame/validate*_split*.pkl > val.list # 将validate*_split*.pkl的路径重新写入val.list
        popd
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain whole train data
        pushd ./data/k400
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/train_link.list
        wget -nc https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list
        bash download_k400_data.sh train_link.list
        bash download_k400_data.sh val_link.list
        # download annotations
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list
        wget -nc https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain whole train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    else
        echo "Not added into TIPC yet."
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    if [ ${model_name} == "PP-TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "PP-TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "STGCN" ]; then
        # pretrain lite train data
        pushd data/fsd10
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_data.npy
        wget -nc https://videotag.bj.bcebos.com/Data/FSD_train_label.npy
        popd
    elif [ ${model_name} == "TSM" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
        tar -xf k400_rawframes_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
        # download pretrained weights
        wget -nc -P ./data https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        # pretrain lite train data
        pushd data/yt8m
        ## download & decompression training data
        wget -nc https://videotag.bj.bcebos.com/Data/yt8m_rawframe_small.tar
        tar -xf yt8m_rawframe_small.tar
        python3.7 -m pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
        python3.7 tf2pkl.py ./frame ./pkl_frame/
        ls pkl_frame/train*.pkl > train_small.list # 将train*.pkl的路径写入train_small.list
        ls pkl_frame/validate*.pkl > val_small.list # 将validate*.pkl的路径写入val_small.list

        python3.7 split_yt8m.py train_small.list # 拆分每个train*.pkl变成多个train*_split*.pkl
        python3.7 split_yt8m.py val_small.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl

        ls pkl_frame/train*_split*.pkl > train_small.list # 将train*_split*.pkl的路径重新写入train_small.list
        ls pkl_frame/validate*_split*.pkl > val_small.list # 将validate*_split*.pkl的路径重新写入val_small.list
        popd
    elif [ ${model_name} == "SlowFast" ]; then
        # pretrain lite train data
        pushd ./data/k400
        wget -nc https://videotag.bj.bcebos.com/Data/k400_videos_small.tar
        tar -xf k400_videos_small.tar
        popd
    elif [ ${model_name} == "BMN" ]; then
        # pretrain lite train data
        pushd ./data
        mkdir bmn_data
        cd bmn_data
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz
        tar -xf bmn_feat.tar.gz
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json
        wget -nc https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json
        popd
    else
        echo "Not added into TIPC yet."
    fi
elif [ ${MODE} = "whole_infer" ];then
    if [ ${model_name} = "PP-TSM" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams --no-check-certificate
    elif [ ${model_name} = "PP-TSN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "AGCN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/AGCN_fsd.pdparams --no-check-certificate
    elif [ ${model_name} == "STGCN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/STGCN_fsd.pdparams --no-check-certificate
    elif [ ${model_name} == "TSM" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "TSN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "TimeSformer" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams --no-check-certificate
    elif [ ${model_name} == "AttentionLSTM" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/AttentionLSTM_yt8.pdparams --no-check-certificate
    elif [ ${model_name} == "SlowFast" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams --no-check-certificate
    elif [ ${model_name} == "BMN" ]; then
        # download pretrained weights
        wget -nc -P ./data https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams --no-check-certificate
    else
        echo "Not added into TIPC yet."
    fi
fi

if [ ${MODE} = "klquant_whole_infer" ]; then
    echo "Not added into TIPC now."
fi

if [ ${MODE} = "cpp_infer" ];then
    # install required packages
    apt-get update
    apt install libavformat-dev
    apt install libavcodec-dev
    apt install libswresample-dev
    apt install libswscale-dev
    apt install libavutil-dev
    apt install libsdl1.2-dev
    apt-get install ffmpeg

    if [ ${model_name} = "PP-TSM" ]; then
        # download pretrained weights
        wget -nc -P data/ https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams --no-check-certificate
        # export inference model
        python3.7 tools/export_model.py -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml -p data/ppTSM_k400_uniform.pdparams -o ./inference/ppTSM
    elif [ ${model_name} = "PP-TSN" ]; then
        # download pretrained weights
        wget -nc -P data/ https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400.pdparams --no-check-certificate
        # export inference model
        python3.7 tools/export_model.py -c configs/recognition/pptsn/pptsn_k400_videos.yaml -p data/ppTSN_k400.pdparams -o ./inference/ppTSN
    else
        echo "Not added into TIPC now."
    fi
fi

if [ ${MODE} = "serving_infer" ];then
    echo "Not added into TIPC now."
fi

if [ ${MODE} = "paddle2onnx_infer" ];then
    echo "Not added into TIPC now."
fi
