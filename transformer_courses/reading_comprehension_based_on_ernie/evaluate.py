import time
import json
import argparse
import collections
import paddle
import paddlenlp
from functools import partial
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.transformers import ErnieForQuestionAnswering
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.datasets import load_dataset
from data_processor import prepare_train_features, prepare_validation_features



def evaluate(args, is_test=True):
    # 加载模型
    model_state = paddle.load(args.model_path)
    model = ErnieForQuestionAnswering.from_pretrained(args.model_name) 
    model.load_dict(model_state)
    model.eval()

    # 加载数据
    train_ds, dev_ds, test_ds = load_dataset('dureader_robust', splits=('train', 'dev', 'test'))
    tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(args.model_name)    
    test_trans_func = partial(prepare_validation_features, 
                           max_seq_length=args.max_seq_length, 
                           doc_stride=args.doc_stride,
                           tokenizer=tokenizer)
    test_ds.map(test_trans_func, batched=True, num_workers=4)
    test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)

    test_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)

    test_data_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=test_batchify_fn,
        return_list=True)
    
    

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in test_data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids,
                                                       token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 10 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, _, _ = compute_prediction(
        test_data_loader.dataset.data, test_data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), False, 20, 30)

    if is_test:
        # Can also write all_nbest_json and scores_diff_json files if needed
        with open('prediction.json', "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    all_predictions, ensure_ascii=False, indent=4) + "\n")
    else:
        squad_evaluate(
            examples=test_data_loader.dataset.data,
            preds=all_predictions,
            is_whitespace_splited=False)

    count = 0
    for example in test_data_loader.dataset.data:
        count += 1
        print()
        print('问题：',example['question'])
        print('原文：',''.join(example['context']))
        print('答案：',all_predictions[example['id']])
        if count >= 5:
            break

    model.train()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Reading Comprehension based on ERNIE.")
    parser.add_argument("--model_name", type=str, default="ernie-1.0", help="the model you want to load.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="the max_seq_length of input sequence.")
    parser.add_argument("--doc_stride", type=int, default=128, help="doc_stride when processing data.")
    parser.add_argument("--batch_size", type=int, default=12, help="batch_size when model training.")
    parser.add_argument("--model_path", type=str, default="./ernie_rc.pdparams", help="the path of saving model.")

    args = parser.parse_args()
    
    evaluate(args)
