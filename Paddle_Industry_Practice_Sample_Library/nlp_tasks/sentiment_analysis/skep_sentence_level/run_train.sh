export CUDA_VISIBLE_DEVICES=0

python  train.py \
        --num_epoch 3 \
        --batch_size 16 \
        --max_seq_len 256 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_step 50 \
        --eval_step 1000 \
        --seed 1000 \
        --device "gpu" \
        --checkpoint "./checkpoint"
