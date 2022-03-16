export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --model_path "./checkpoint/best.pdparams" \
        --ori_schema_path "./data/duie_schema.json" \
        --save_label_path "./data/label.dict" \
        --save_schema_path "./data/schema.json" \
        --max_seq_len 512


