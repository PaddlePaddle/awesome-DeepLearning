export CUDA_VISIBLE_DEVICES=0

python  train.py \
        --train_path "./data/duie_train.json" \
        --dev_path "./data/duie_dev.json" \
        --ori_schema_path "./data/duie_schema.json" \
        --save_label_path "./data/label.dict" \
        --save_schema_path "./data/schema.json" \
        --num_epoch 10 \
        --batch_size 32 \
        --max_seq_len 512 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_step 50 \
        --eval_step 1000 \
        --seed 1000 \
        --device "gpu" \
        --checkpoint "./checkpoint"


