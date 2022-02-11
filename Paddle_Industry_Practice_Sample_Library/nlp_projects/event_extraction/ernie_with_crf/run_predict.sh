export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --trigger_model_path "./checkpoint/trigger_best.pdparams" \
        --role_model_path "./checkpoint/role_best.pdparams" \
        --trigger_tag_path "./data/dict/trigger.dict" \
        --role_tag_path "./data/dict/role.dict" \
        --schema_path "./data/duee_event_schema.json"
