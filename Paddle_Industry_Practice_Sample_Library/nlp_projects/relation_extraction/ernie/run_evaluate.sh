export CUDA_VISIBLE_DEVICES=0

python  evaluate.py \
        --model_path "./checkpoint/best.pdparams" \
        --test_path "./data/duie_dev.json" \
        --ori_schema_path "./data/duie_schema.json" \
        --save_label_path "./data/label.dict" \
        --save_schema_path "./data/schema.json" \
        --batch_size 32 \
        --max_seq_len 512


