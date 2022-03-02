export CUDA_VISIBLE_DEVICES=0

model=$1

if [ ${model} == trigger ]; then
    echo "============> start to evaluate trigger model <============"
    python  evaluate.py \
        --model_name "trigger" \
        --model_path "./checkpoint/trigger_best.pdparams" \
        --dev_path "./data/trigger/duee_dev.tsv" \
        --tag_path "./data/dict/trigger.dict"
elif [ ${model} == role ]; then
    echo "============> start to evaluate role model <============"
    python  evaluate.py \
        --model_name "role" \
        --model_path "./checkpoint/role_best.pdparams" \
        --dev_path "./data/role/duee_dev.tsv" \
        --tag_path "./data/dict/role.dict"
else
    echo "============> start to evaluate trigger model <============"
    python  evaluate.py \
        --model_name "trigger" \
        --model_path "./checkpoint/trigger_best.pdparams" \
        --dev_path "./data/trigger/duee_dev.tsv" \
        --tag_path "./data/dict/trigger.dict"
    echo "============> start to evaluate role model <============"
    python  evaluate.py \
        --model_name "role" \
        --model_path "./checkpoint/role_best.pdparams" \
        --dev_path "./data/role/duee_dev.tsv" \
        --tag_path "./data/dict/role.dict"
fi
