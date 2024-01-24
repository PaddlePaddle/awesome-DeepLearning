# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪(TODO)、量化(TODO)、蒸馏。

- Mac端基础训练预测功能测试参考[TODO]()
- Windows端基础训练预测功能测试参考[TODO]()

## 1. 测试结论汇总

- 训练相关：

    | 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
    |  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
    |  PP-TSM  | pptsm_k400_frames_uniform | 正常训练 | 正常训练 | - | - |
    |  PP-TSN  | pptsn_k400_videos | 正常训练 | 正常训练 | - | - |
    |  AGCN  | agcn_fsd | 正常训练 | 正常训练 | - | - |
    |  STGCN  | stgcn_fsd | 正常训练 | 正常训练 | - | - |
    |  TimeSformer  | timesformer_k400_videos | 正常训练 | 正常训练 | - | - |
    |  SlowFast  | slowfast | 正常训练 | 正常训练 | - | - |
    |  TSM  | tsm_k400_frames | 正常训练 | 正常训练 | - | - |
    |  TSN  | tsn_k400_frames | 正常训练 | 正常训练 | - | - |
    |  AttentionLSTM  | attention_lstm_youtube8m | 正常训练 | 正常训练 | - | - |
    |  BMN  | bmn | 正常训练 | 正常训练 | - | - |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型(TODO)`，这两类模型对应的预测功能汇总如下，

    | 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
    |  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
    | 正常模型 | GPU | 1/2 | fp32/fp16 | - | 1/6 |
    | 正常模型 | CPU | 1/2 | - | fp32/fp16 | 1/6 |


## 2. 测试流程

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 安装依赖
- 安装对应软硬件环境下的PaddlePaddle（>=2.0）

- 安装PaddleVideo依赖
    ```
    # 需在PaddleVideo目录下执行
    python3.7 -m pip install -r requirements.txt
    ```
- 安装autolog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    python3.7 -m pip install -r requirements.txt
    python3 setup.py bdist_wheel
    python3.7 -m pip install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```
- 安装PaddleSlim (可选)
   ```
   # 如果要测试量化、裁剪等功能，则需用以下命令安装PaddleSlim
   python3.7 -m pip install paddleslim
   ```


### 2.2 基本功能测试
1. 先运行`prepare.sh`，根据传入模型名字，准备对应数据和预训练模型参数
2. 再运行`test_train_inference_python.sh`，根据传入模型名字，进行对应测试
3. 在`test_tipc/output`目录下生成 `python_infer_*.log` 格式的日志文件

具体地，以PP-TSM的测试链条为例，运行细节如下：

`test_train_inference_python.sh` 包含5种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：**lite_train_lite_infer**，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
    ```shell
    bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'lite_train_lite_infer'
    bash test_tipc/test_train_inference_python.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'lite_train_lite_infer'
    ```

- 模式2：**lite_train_whole_infer**，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
    ```shell
    bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt  'lite_train_whole_infer'
    bash test_tipc/test_train_inference_python.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'lite_train_whole_infer'
    ```

- 模式3：**whole_infer**，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度；
    ```shell
    bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'whole_infer'
    # 用法1:
    bash test_tipc/test_train_inference_python.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'whole_infer'
    # 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
    bash test_tipc/test_train_inference_python.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'whole_infer' '1'
    ```

- 模式4：**whole_train_whole_infer**： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
    ```shell
    bash test_tipc/prepare.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'whole_train_whole_infer'
    bash test_tipc/test_train_inference_python.sh test_tipc/configs/PP-TSM/PP-TSM_train_infer_python.txt 'whole_train_whole_infer'
    ```


最终在`tests/output/model_name`目录下生成.log后缀的日志文件


### 2.3 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取`*.log`日志中的预测结果，包括类别和概率
- 从本地文件中提取保存好的真值结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

#### 使用方式
运行命令：
```shell
python3.7 test_tipc/compare_results.py --gt_file="test_tipc/results/python_*.txt"  --log_file="test_tipc/output/python_*.log" --atol=1e-3 --rtol=1e-3
```

参数介绍：  
- gt_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在test_tipc/result/ 文件夹下
- log_file: 指向运行test_tipc/test_train_inference_python.sh 脚本的infer模式保存的预测日志，预测日志中打印的有预测结果，比如：预测文本，类别等等，同样支持python_infer_*.log格式传入
- atol: 设置的绝对误差
- rtol: 设置的相对误差

#### 运行结果

正常运行效果如下：
```bash
Assert allclose passed! The results of python_infer_cpu_usemkldnn_False_threads_6_precision_fp32_batchsize_16.log and ./test_tipc/results/PP-TSM/python_ppvideo_PP-TSM_results_fp32.txt are consistent!
```

出现不一致结果时的样例输出如下：
```bash
ValueError: The results of python_infer_gpu_usetrt_False_precision_fp32_batchsize_8.log and the results of ./test_tipc/results/PP-TSM/python_ppvideo_PP-TSM_results_fp32.txt are inconsistent!
```
