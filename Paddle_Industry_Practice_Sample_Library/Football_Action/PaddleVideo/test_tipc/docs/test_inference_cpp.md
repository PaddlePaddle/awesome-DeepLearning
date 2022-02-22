# C++预测功能测试

C++预测功能测试的主程序为`test_inference_cpp.sh`，可以测试基于C++预测库的模型推理功能。

## 1. 测试结论汇总

基于训练是否使用量化，进行本测试的模型可以分为`正常模型`和`量化模型`(TODO)，这两类模型对应的C++预测功能汇总如下：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1/6 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1/6 | - | fp32 | 支持 |

## 2. 测试流程
运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_inference_cpp.sh`进行测试，最终在```test_tipc/output```目录下生成`cpp_infer_*.log`后缀的日志文件。

```bash
bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/PP-TSM_infer_cpp.txt 'cpp_infer'
```
```bash
# 用法1:
bash test_tipc/test_inference_cpp.sh test_tipc/configs/PP-TSM/PP-TSM_infer_cpp.txt
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_inference_cpp.sh test_tipc/configs/PP-TSM/PP-TSM_infer_cpp.txt 1
```

运行预测指令后，在`test_tipc/output`文件夹下自动会保存运行日志，包括以下文件：

```shell
test_tipc/PP-TSM/output/
    ├── results_cpp.log    # 运行指令状态的日志
    ├── cpp_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log  # CPU上不开启Mkldnn，线程数设置为1，测试batch_size=1条件下的预测运行日志
    ├── cpp_infer_cpu_usemkldnn_False_threads_6_precision_fp32_batchsize_1.log  # CPU上不开启Mkldnn，线程数设置为6，测试batch_size=1条件下的预测运行日志
    ├── cpp_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log # GPU上不开启TensorRT，测试batch_size=1的fp32精度预测日志
    ├── cpp_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log  # GPU上开启TensorRT，测试batch_size=1的fp16精度预测日志
......
```
其中results_cpp.log中包含了每条指令的运行状态，如果运行成功会输出：

```
Run successfully with command - ./deploy/cpp_infer/build/ppvideo rec --use_gpu=True --use_tensorrt=False --precision=fp32 --rec_model_dir=./inference/ppTSM --rec_batch_num=1 --video_dir=./deploy/cpp_infer/example_video_dir --benchmark=True --inference_model_name=ppTSM --char_list_file=data/k400/Kinetics-400_label_list.txt --num_seg=8 > ./test_tipc/output/PP-TSM/cpp_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log 2>&1
......
```
如果运行失败，会输出：
```
Run failed with command - ./deploy/cpp_infer/build/ppvideo rec --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --rec_model_dir=./inference/ppTSM --rec_batch_num=1 --video_dir=./deploy/cpp_infer/example_video_dir --benchmark=True --inference_model_name=ppTSM --char_list_file=data/k400/Kinetics-400_label_list.txt --num_seg=8 > ./test_tipc/output/PP-TSM/cpp_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log 2>&1
......
```
可以很方便的根据results_cpp.log中的内容判定哪一个指令运行错误。


### 2.2 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取预测输出文本的结果
- 提取本地参考输出文本结果
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

#### 使用方式
运行命令：
```shell
python3.7 test_tipc/compare_results.py --gt_file "test_tipc/results/PP-TSM_CPP/cpp_ppvideo_PP-TSM_results_fp*.txt" --log_file "test_tipc/output/PP-TSM/cpp_infer_*.log" --atol=1e-3 --rtol=1e-3
```

参数介绍：
- gt_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在test_tipc/result/ 文件夹下
- log_file: 指向运行test_tipc/test_inference_cpp.sh 脚本的infer模式保存的预测日志，预测日志中打印的有预测结果，比如：文本框，预测文本，类别等等，同样支持cpp_infer_*.log格式传入
- atol: 设置的绝对误差
- rtol: 设置的相对误差

#### 运行结果

正常运行输出示例：
```bash
Assert allclose passed! The results of cpp_infer_cpu_usemkldnn_True_threads_1_precision_fp32_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp32.txt are consistent!
Assert allclose passed! The results of cpp_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp32.txt are consistent!
Assert allclose passed! The results of cpp_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp16.txt are consistent!
Assert allclose passed! The results of cpp_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp32.txt are consistent!
Assert allclose passed! The results of cpp_infer_cpu_usemkldnn_True_threads_6_precision_fp32_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp32.txt are consistent!
Assert allclose passed! The results of cpp_infer_cpu_usemkldnn_False_threads_6_precision_fp32_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp32.txt are consistent!
Assert allclose passed! The results of cpp_infer_gpu_usetrt_True_precision_fp32_batchsize_1.log and test_tipc/results/PP-TSN_CPP/cpp_ppvideo_PP-TSN_results_fp32.txt are consistent!
```

出现不一致结果时的运行输出示例：
```bash
ValueError: The results of cpp_infer_cpu_usemkldnn_True_threads_1_precision_fp32_batchsize_1.log and the results of test_tipc/results/PP-TSM_CPP/cpp_ppvideo_PP-TSM_results_fp32.txt are inconsistent!
```


## 3. 更多教程

本文档为功能测试用，更详细的C++预测使用教程请参考：[服务器端C++预测](../../deploy/cpp_infer/readme.md)  
