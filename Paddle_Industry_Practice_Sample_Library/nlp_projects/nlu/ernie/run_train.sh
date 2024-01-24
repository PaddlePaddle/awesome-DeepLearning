export CUDA_VISIBLE_DEVICES=0

python  train.py \
        --train_path "./data/train.json" \
        --dev_path "./data/test.json" \
        --intent_dict_path "./data/intent_labels.json" \
        --slot_dict_path "./data/slot_labels.json" \
        --learning_rate 3e-5 \
        --num_epoch 10 \
        --batch_size 32 \
        --warmup_proportion 0.01 \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --eval_step 1000 \
        --use_history false \
        --checkpoint "./checkpoint"


