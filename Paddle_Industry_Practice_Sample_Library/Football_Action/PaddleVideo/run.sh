export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#export FLAGS_conv_workspace_size_limit=800 #MB
#export FLAGS_cudnn_exhaustive_search=1
#export FLAGS_cudnn_batchnorm_spatial_persistent=1


start_time=$(date +%s)

# run ava training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=logdir.ava_part main.py --validate -w paddle.init_param.pdparams -c configs/detection/ava/ava_part.yaml
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logdir.ava_all.1203 main.py --validate -w paddle.init_param.pdparams -c configs/detection/ava/ava_all.yaml

# run adds training
# python3.7 main.py --validate -c configs/estimation/adds/adds.yaml --seed 20

# run tsm training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_k400_frames.yaml

# run tsm amp training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_tsm main.py  --amp --validate -c configs/recognition/tsm/tsm_k400_frames.yaml

# run tsm amp training, nhwc
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_tsm main.py  --amp --validate -c configs/recognition/tsm/tsm_k400_frames_nhwc.yaml

# run tsn training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_tsn main.py  --validate -c configs/recognition/tsn/tsn_k400_frames.yaml

# run video-swin-transformer training
# python3.7 -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_videoswin main.py --amp --validate -c configs/recognition/videoswin/videoswin_k400_videos.yaml

# run slowfast training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast  main.py --validate -c configs/recognition/slowfast/slowfast.yaml

# run slowfast multi-grid training
# python3.7 -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=log-slowfast main.py --validate --multigrid -c configs/recognition/slowfast/slowfast_multigrid.yaml

# run bmn training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_bmn main.py  --validate -c configs/localization/bmn.yaml

# run attention_lstm training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  --validate -c configs/recognition/attention_lstm/attention_lstm_youtube-8m.yaml

# run pp-tsm training
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml

# run pp-tsn training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsn  main.py  --validate -c configs/recognition/pptsn/pptsn_k400_frames.yaml

# run timesformer training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_timesformer  main.py  --validate -c configs/recognition/timesformer/timesformer_k400_videos.yaml

# run pp-timesformer training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer  main.py  --validate -c configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml

# run st-gcn training
# python3.7 main.py -c configs/recognition/stgcn/stgcn_fsd.yaml

# run agcn training
# python3.7 main.py -c configs/recognition/agcn/agcn_fsd.yaml

# run actbert training
# python3.7 main.py  --validate -c configs/multimodal/actbert/actbert.yaml

# run tsn dali training
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsn main.py --train_dali -c configs/recognition/tsn/tsn_dali.yaml


# test.sh
# just use `example` as example, please replace to real name.
# python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_test main.py --test -c configs/example.yaml -w "output/example/example_best.pdparams"

# NOTE: run bmn test, only support single card, bs=1
# python3.7 main.py --test -c configs/localization/bmn.yaml -w output/BMN/BMN_epoch_00010.pdparams -o DATASET.batch_size=1

# export_models script
# just use `example` as example, please replace to real name.
# python3.7 tools/export_model.py -c configs/example.yaml -p output/example/example_best.pdparams -o ./inference

# predict script
# just use `example` as example, please replace to real name.
# python3.7 tools/predict.py -v example.avi --model_file "./inference/example.pdmodel" --params_file "./inference/example.pdiparams" --enable_benchmark=False --model="example" --num_seg=8

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Time to train is $(($cost_time/60))min $(($cost_time%60))s"
