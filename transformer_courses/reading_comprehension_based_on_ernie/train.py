import paddle
import paddlenlp
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieForQuestionAnswering
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
import time
import argparse
from functools import partial
from utils import CrossEntropyLossForRobust
from data_processor import prepare_train_features, prepare_validation_features

def evaluate(model, data_loader, is_test=False):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in data_loader:
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
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), False, 20, 30)

    if is_test:
        # Can also write all_nbest_json and scores_diff_json files if needed
        with open('prediction.json', "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    all_predictions, ensure_ascii=False, indent=4) + "\n")
    else:
        squad_evaluate(
            examples=data_loader.dataset.data,
            preds=all_predictions,
            is_whitespace_splited=False)

    count = 0
    for example in data_loader.dataset.data:
        count += 1
        print()
        print('问题：',example['question'])
        print('原文：',''.join(example['context']))
        print('答案：',all_predictions[example['id']])
        if count >= 5:
            break

    model.train()

def train(args):
    
    # 加载数据集
    train_ds, dev_ds, test_ds = load_dataset('dureader_robust', splits=('train', 'dev', 'test'))

    tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(args.model_name)

    train_trans_func = partial(prepare_train_features, 
                           max_seq_length=args.max_seq_length, 
                           doc_stride=args.doc_stride,
                           tokenizer=tokenizer)

    train_ds.map(train_trans_func, batched=True, num_workers=4)

    dev_trans_func = partial(prepare_validation_features, 
                           max_seq_length=args.max_seq_length, 
                           doc_stride=args.doc_stride,
                           tokenizer=tokenizer)
                           
    dev_ds.map(dev_trans_func, batched=True, num_workers=4)
    test_ds.map(dev_trans_func, batched=True, num_workers=4)


    # 定义BatchSampler
    train_batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)

    # 定义batchify_fn
    train_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "start_positions": Stack(dtype="int64"),
        "end_positions": Stack(dtype="int64")
        }): fn(samples)

    dev_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)

    # 构造DataLoader
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)

    test_data_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)



    # 训练配置相关
    num_training_steps = len(train_data_loader) * args.epochs
    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')    

    lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)
    
    model = ErnieForQuestionAnswering.from_pretrained(args.model_name)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)


    # 训练代码
    model.train()
    criterion = CrossEntropyLossForRobust()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            global_step += 1
            input_ids, segment_ids, start_positions, end_positions = batch
            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = criterion(logits, (start_positions, end_positions))

            if global_step % 100 == 0 :
                print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        paddle.save(model.state_dict(), args.save_model_path)
        paddle.save(model.state_dict(), args.save_opt_path)
        evaluate(model=model, data_loader=dev_data_loader)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Reading Comprehension based on ERNIE.")
    parser.add_argument("--model_name", type=str, default="ernie-1.0", help="the model you want to load.")
    parser.add_argument("--epochs", type=int, default=2, help="the epochs of model training.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="the max_seq_length of input sequence.")
    parser.add_argument("--doc_stride", type=int, default=128, help="doc_stride when processing data.")
    parser.add_argument("--batch_size", type=int, default=12, help="batch_size when model training.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning_rate for model training.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="the proportion of performing warmup in all training steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="the weight_decay of model parameters.")   
    parser.add_argument("--save_model_path", type=str, default="./ernie_rc.pdparams", help="the path of saving model.") 
    parser.add_argument("--save_opt_path", type=str, default="./ernie_rc.pdopt", help="the path of saving optimizer")

    args = parser.parse_args()

    
    train(args)
