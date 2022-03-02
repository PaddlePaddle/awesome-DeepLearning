export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --model_path "./checkpoint/best.pdparams" \
        --intent_dict_path "./data/intent_labels.json" \
        --slot_dict_path "./data/slot_labels.json" \
        --use_history false 

