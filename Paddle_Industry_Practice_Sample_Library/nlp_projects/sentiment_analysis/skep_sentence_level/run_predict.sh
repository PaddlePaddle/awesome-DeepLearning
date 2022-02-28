export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --model_path "./checkpoint/best.pdparams" \
        --max_seq_len 256


