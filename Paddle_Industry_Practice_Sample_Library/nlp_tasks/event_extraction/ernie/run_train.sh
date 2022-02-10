export CUDA_VISIBLE_DEVICES=0

model=$1

if [ ${model} == trigger ]; then
    echo "============> start to train trigger model <============"
    python  train.py \
        --model_name "trigger" \
        --train_path "./data/trigger/duee_train.tsv" \
        --dev_path "./data/trigger/duee_dev.tsv" \
        --tag_path "./data/dict/trigger.dict" \
        --learning_rate 5e-5 \
        --num_epoch 20 \
        --batch_size 16 \
        --checkpoint "./checkpoint"
elif [ ${model} == role ]; then
    echo "============> start to train role  model <============"
    python  train.py \
        --model_name "role" \
        --train_path "./data/role/duee_train.tsv" \
        --dev_path "./data/role/duee_dev.tsv" \
        --tag_path "./data/dict/role.dict" \
        --learning_rate 5e-5 \
        --num_epoch 20 \
        --batch_size 32 \
        --checkpoint "./checkpoint"
else
    echo "============> start to train trigger model <============"
    python  train.py \
        --model_name "trigger" \
        --train_path "./data/trigger/duee_train.tsv" \
        --dev_path "./data/trigger/duee_dev.tsv" \
        --tag_path "./data/dict/trigger.dict" \
        --learning_rate 5e-5 \
        --num_epoch 20 \
        --batch_size 16 \
        --checkpoint "./checkpoint"
    echo "============> start to train role  model <============"
    python  train.py \
        --model_name "role" \
        --train_path "./data/role/duee_train.tsv" \
        --dev_path "./data/role/duee_dev.tsv" \
        --tag_path "./data/dict/role.dict" \
        --learning_rate 5e-5 \
        --num_epoch 20 \
        --batch_size 32 \
        --checkpoint "./checkpoint"
fi
