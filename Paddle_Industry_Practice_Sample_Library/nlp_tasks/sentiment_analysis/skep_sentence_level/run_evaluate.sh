export CUDA_VISIBLE_DEVICES=0

python  evaluate.py \
        --model_path "./checkpoint/best.pdparams" \
        --test_set "dev" \
        --batch_size 16 \
        --max_seq_len 256

