#! /bin/bash
export FLAGS_eager_delete_tensor_gb=0.0 #  一旦不再使用立即释放内存垃圾
export FLAGS_fast_eager_deletion_mode=1 # 启用快速垃圾回收策略
export FLAGS_fraction_of_gpu_memory_to_use=0
export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=1
#export CUDA_VISIBLE_DEVICES=3
export CPU_NUM=8
ERNIE_PRETRAIN=./senta_model/ernie_pretrain_model
DATA_PATH=./senta_data
MODEL_SAVE_PATH=./save_models/ernie_model

# run_train
train() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train true \
        --do_val true \
        --do_infer false \
        --use_paddle_hub false \
        --batch_size 24 \
        --init_checkpoint $ERNIE_PRETRAIN/params \
        --train_set $DATA_PATH/train.tsv \
        --dev_set $DATA_PATH/dev.tsv \
        --test_set $DATA_PATH/test.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --checkpoints $MODEL_SAVE_PATH \
        --save_steps 5000 \
        --validation_steps 100 \
        --epoch 10 \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --lr 5e-5 \
        --skip_steps 10 \
        --num_labels 2 \
        --random_seed 1
}

# run_eval
evaluate() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train false \
        --do_val true \
        --do_infer false \
        --use_paddle_hub false \
        --batch_size 24 \
        --init_checkpoint ./save_models/step_5000/ \
        --dev_set $DATA_PATH/dev.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --num_labels 2
    
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train false \
        --do_val true \
        --do_infer false \
        --use_paddle_hub false \
        --batch_size 24 \
        --init_checkpoint ./save_models/step_5000/ \
        --dev_set $DATA_PATH/test.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --num_labels 2
}

# run_infer
infer() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train false \
        --do_val false \
        --do_infer true \
        --use_paddle_hub false \
        --batch_size 24 \
        --init_checkpoint ./save_models/step_5000 \
        --test_set $DATA_PATH/test.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --num_labels 2
}

main() {
    local cmd=${1:-help}
    case "${cmd}" in
        train)
            train "$@";
            ;;
        eval)
            evaluate "$@";
            ;;
        infer)
            infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 0;
            ;;
        *)
            echo "Unsupport commend [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 1;
            ;;
    esac
}
main "$@"
