
home_dir="/home/work"
save_dir="$home_dir/checkpoints/models"
if [ ! -d "$save_dir" ]; then
    mkdir "$save_dir"
fi

# ppTSM v2.0
cd $home_dir/PaddleVideo-release-2.0
python -B -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    --log_dir=$save_dir/logs \
    main.py  \
    --validate \
    -c configs/recognition/tsm/pptsm_football.yaml \
    -o output_dir=$save_dir

# # BMN v2.0
# cd $home_dir/PaddleVideo-release-2.0
# cd $home_dir/train_proposal
# python -B -m paddle.distributed.launch \
#     --gpus="0,1,2,3" \
#     --log_dir=$save_dir/logs \
#     main.py  \
#     --validate \
#     -c configs/localization/bmn_football.yaml \
#     -o output_dir=$save_dir

# # BMN v1.8
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd $home_dir/train_proposal
# python -u train.py \
#         --model_name=BMN \
#         --config=$work_dir/configs/bmn_football_v1.8.yaml \
#         --save_dir=$save_dir \
#         2>&1 | tee $LOG

# # LSTM v1.8
# export CUDA_VISIBLE_DEVICES=0,1
# cd $home_dir/train_lstm
# LOG="$save_dir/log_train"
# python -u scenario_lib/train.py \
#     --model_name=ActionNet \
#     --config=$work_dir/conf/conf.txt \
#     --save_dir=$save_dir \
#     --log_interval=5 \
#     --valid_interval=1 \
#     2>&1 | tee $LOG

# # export
# cd $home_dir/PaddleVideo-release-2.0
# python3 tools/export_model.py -c configs/recognition/tsm/pptsm_football.yaml \
#                               -p /home/work/checkpoints/models_pptsm_pp/ppTSM_epoch_00057.pdparams \
#                               -o /home/work/inference/checkpoints/ppTSM

