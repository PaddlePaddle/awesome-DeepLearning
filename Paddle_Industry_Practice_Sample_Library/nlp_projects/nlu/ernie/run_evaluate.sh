export CUDA_VISIBLE_DEVICES=0

python  evaluate.py \
        --model_path "./checkpoint/best.pdparams" \
        --test_path "./data/test.json" \
        --intent_dict_path "./data/intent_labels.json" \
        --slot_dict_path "./data/slot_labels.json" \
        --batch_size 20 \
        --use_history false


