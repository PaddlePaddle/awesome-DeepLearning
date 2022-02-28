[English](./readme_en.md) | 简体中文

# 服务器端C++预测

本章节介绍PaddleVideo模型的的C++部署方法，python预测部署方法请参考各自模型的**模型推理**章节。
C++在性能计算上优于python，因此，在大多数CPU、GPU部署场景，多采用C++的部署方式，本节将介绍如何在Linux（CPU/GPU）环境下配置C++环境并完成
PaddleVideo模型部署。

## 1. 准备环境

- Linux环境，推荐使用docker。

- Windows环境，目前支持基于`Visual Studio 2019 Community`进行编译（TODO）

* 该文档主要介绍基于Linux环境的PaddleVideo C++预测流程，如果需要在Windows下基于预测库进行C++预测，具体编译方法请参考[Windows下编译教程](./docs/windows_vs2019_build.md)（TODO）
* **准备环境的目的是得到编译好的opencv库与paddle预测库**。

### 1.1 编译opencv库

* 首先需要从opencv官网上下载在Linux环境下源码编译的压缩包，并解压成文件夹。以opencv3.4.7为例，下载命令如下：

    ```bash
    cd deploy/cpp_infer
    wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
    tar -xf 3.4.7.tar.gz
    ```

    解压完毕后在`deploy/cpp_infer`目录下可以得到解压出的`opencv-3.4.7`的文件夹。

* 安装ffmpeg

    opencv配合ffmpeg才能在linux下正常读取视频，否则可能遇到视频帧数返回为0或无法读取任何视频帧的情况

    采用较为简单的apt安装，安装命令如下：

    ```bash
    apt-get update

    apt install libavformat-dev
    apt install libavcodec-dev
    apt install libswresample-dev
    apt install libswscale-dev
    apt install libavutil-dev
    apt install libsdl1.2-dev

    apt-get install ffmpeg
    ```

* 准备编译opencv，首先进入`opencv-3.4.7`的文件夹，然后设置opencv源码路径`root_path`以及安装路径`install_path`。执行命令如下：

    ```bash
    cd opencv-3.4.7

    root_path=/xxx/xxx/xxx/xxx/opencv-3.4.7 # 填写为刚解压出来的opencv-3.4.7绝对路径
    install_path=${root_path}/opencv3

    rm -rf build
    mkdir build
    cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${install_path} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWITH_IPP=OFF \
        -DBUILD_IPP_IW=OFF \
        -DWITH_LAPACK=OFF \
        -DWITH_EIGEN=OFF \
        -DCMAKE_INSTALL_LIBDIR=lib64 \
        -DWITH_ZLIB=ON \
        -DBUILD_ZLIB=ON \
        -DWITH_JPEG=ON \
        -DBUILD_JPEG=ON \
        -DWITH_PNG=ON \
        -DBUILD_PNG=ON \
        -DWITH_TIFF=ON \
        -DBUILD_TIFF=ON \
        -DWITH_FFMPEG=ON

    make -j
    make install
    ```

    `make install`完成之后，会在该文件夹下生成opencv头文件和库文件，用于后面的Video推理C++代码编译。

    最终会以安装路径`install_path`为指定路径，得到一个`opencv3`的文件夹，其文件结构如下所示。

    ```
    opencv3/
    ├── bin/
    ├── include/
    ├── lib/
    ├── lib64/
    └── share/
    ```

### 1.2 下载或者编译Paddle预测库

有2种方式获取Paddle预测库，下面进行详细介绍。


#### 1.2.1 直接下载安装

* [Paddle预测库官网](https://paddleinference.paddlepaddle.org.cn/v2.1/user_guides/download_lib.html) 上提供了不同cuda版本的Linux预测库，可以在官网查看并**选择合适的预测库版本**（建议选择paddle版本>=2.0.1版本的预测库）。

* 下载得到一个`paddle_inference.tgz`压缩包，然后将它解压成文件夹，命令如下(以机器环境为gcc8.2为例)：

    ```bash
    wget https://paddle-inference-lib.bj.bcebos.com/2.1.1-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddle_inference.tgz
    tar -xf paddle_inference.tgz
    ```

    最终会在当前的文件夹中生成`paddle_inference/`的子文件夹。

#### 1.2.2 预测库源码编译
* 如果希望获取最新预测库特性，可以从Paddle github上克隆最新代码，源码编译预测库。
* 可以参考[Paddle预测库安装编译说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#congyuanmabianyi) 的说明，从github上获取Paddle代码，然后进行编译，生成最新的预测库。使用git获取代码方法如下。

    ```shell
    git clone https://github.com/PaddlePaddle/Paddle.git
    git checkout release/2.1
    ```

* 进入Paddle目录后，编译方法如下。

    ```shell
    rm -rf build
    mkdir build
    cd build

    cmake  .. \
        -DWITH_CONTRIB=OFF \
        -DWITH_MKL=ON \
        -DWITH_MKLDNN=ON  \
        -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_INFERENCE_API_TEST=OFF \
        -DON_INFER=ON \
        -DWITH_PYTHON=ON
    make -j4
    make inference_lib_dist -j4 # 4为编译时使用核数，可根据机器情况自行修改
    ```

    更多编译参数选项介绍可以参考[文档说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#congyuanmabianyi)。


* 编译完成之后，可以在`build/paddle_inference_install_dir/`文件下看到生成了以下文件及文件夹。

    ```bash
    build/paddle_inference_install_dir/
    ├── CMakeCache.txt
    ├── paddle/
    ├── third_party
    └── version.txt
    ```

    其中`paddle`就是C++预测所需的Paddle库，`version.txt`中包含当前预测库的版本信息。

## 2. 编译并运行预测demo

### 2.1 将模型导出为inference model

* 该步骤与python部署方式下的导出预测模型相同，可以参考各自模型的模型预测章节。导出的几个相关inference model文件用于模型预测。**以PP-TSM为例**，导出预测模型的目录结构如下。

    ```
    inference/
    └── ppTSM/
    	├── ppTSM.pdiparams
    	├── ppTSM.pdiparamsinfo
    	└── ppTSM.pdmodel
    ```


### 2.2 编译PaddleVideo C++预测demo

* 进入到`deploy/cpp_infer`目录下，执行以下编译命令

    ```shell
    bash tools/build.sh
    ```

    `tools/build.sh`中的Paddle C++预测库、opencv等其他依赖库的地址需要换成自己机器上的实际地址。

* 具体的，需要修改`tools/build.sh`中环境路径，相关内容如下：

    ```shell
    OPENCV_DIR=your_opencv_dir
    LIB_DIR=your_paddle_inference_dir
    CUDA_LIB_DIR=your_cuda_lib_dir
    CUDNN_LIB_DIR=your_cudnn_lib_dir
    TENSORRT_DIR=your_tensorRT_dir
    ```

    以PP-TSM为例，上述参数如下(xxx部分根据用户自己机器情况对应修改)

    ```bash
    OPENCV_DIR=/xxx/xxx/xxx/xxx/xxx/xxx/opencv3
    LIB_DIR=/xxx/xxx/xxx/xxx/xxx/paddle_inference
    CUDA_LIB_DIR=/xxx/xxx/cuda-xxx/lib64
    CUDNN_LIB_DIR=/xxx/xxx/cuda-xxx/lib64
    TENSORRT_DIR=/xxx/xxx/TensorRT-7.0.0.11
    ```

    其中，`OPENCV_DIR`为opencv编译安装的地址；`LIB_DIR`为下载(`paddle_inference`文件夹)或者编译生成的Paddle预测库地址(`build/paddle_inference_install_dir`文件夹)；`CUDA_LIB_DIR`为cuda库文件地址，在docker中为`/usr/local/cuda/lib64`；`CUDNN_LIB_DIR`为cudnn库文件地址，在docker中为`/usr/lib/x86_64-linux-gnu/`。**注意：以上路径都写绝对路径，不要写相对路径。**


* 编译完成之后，会在`cpp_infer/build`文件夹下生成一个名为`ppvideo`的可执行文件。


### 2.3 运行PaddleVideo C++预测demo

运行方式：

```bash
./build/ppvideo <mode> [--param1] [--param2] [...]
```

其中，`mode`为必选参数，表示选择的功能，取值范围['rec']，表示**视频识别**（更多功能会陆续加入）。

##### 1. 调用视频识别：
```bash
# 调用PP-TSM识别
./build/ppvideo rec \
    --rec_model_dir=../../inference/ppTSM \
    --inference_model_name=ppTSM \
    --video_dir=./example_video_dir \
    --num_seg=8 \
    --seg_len=1

# 调用PP-TSN识别
./build/ppvideo rec \
    --rec_model_dir=../../inference/ppTSN \
    --inference_model_name=ppTSN \
    --video_dir=./example_video_dir \
    --num_seg=25 \
    --seg_len=1
```
更多参数如下：

- 通用参数

    | 参数名称      | 类型 | 默认参数        | 意义                                                         |
    | ------------- | ---- | --------------- | ------------------------------------------------------------ |
    | use_gpu       | bool | false           | 是否使用GPU                                                  |
    | gpu_id        | int  | 0               | GPU id，使用GPU时有效                                        |
    | gpu_mem       | int  | 4000            | 申请的GPU内存                                                |
    | cpu_threads   | int  | 10              | CPU预测时的线程数，在机器核数充足的情况下，该值越大，预测速度越快 |
    | enable_mkldnn | bool | false           | 是否使用mkldnn库                                             |
    | use_tensorrt  | bool | false           | 是否使用tensorrt库                                           |
    | precision     | str  | "fp32"          | 使用fp32/fp16/uint8精度来预测                                |
    | benchmark     | bool | true            | 预测时是否开启benchmark，开启后会在最后输出配置、模型、耗时等信息。 |
    | save_log_path | str  | "./log_output/" | 预测结果保存目录                                             |


- 视频识别模型相关

    | 参数名称       | 类型   | 默认参数                                      | 意义                                 |
    | -------------- | ------ | --------------------------------------------- | ------------------------------------ |
    | video_dir      | string | "../example_video_dir"                        | 存放将要识别的视频的文件夹路径       |
    | rec_model_dir  | string | ""                                            | 存放导出的预测模型的文件夹路径       |
    | inference_model_name | string | "ppTSM"                                 | 预测模型的名称 |
    | num_seg        | int    | 8                                             | 视频分段的段数                       |
    | seg_len        | int    | 1                                             | 视频每段抽取的帧数                   |
    | rec_batch_num  | int    | 1                                             | 模型预测时的batch size               |
    | char_list_file | str    | "../../data/k400/Kinetics-400_label_list.txt" | 存放所有类别标号和对应名字的文本路径 |

​	以example_video_dir下的样例视频`example01.avi`为输入视频为例，最终屏幕上会输出检测结果如下。

```bash
[./inference/ppTSM]
[./deploy/cpp_infer/example_video_dir]
total videos num: 1
./example_video_dir/example01.avi   class: 5 archery       score: 0.999556
I1125 08:10:45.834288 13955 autolog.h:50] ----------------------- Config info -----------------------
I1125 08:10:45.834458 13955 autolog.h:51] runtime_device: cpu
I1125 08:10:45.834467 13955 autolog.h:52] ir_optim: True
I1125 08:10:45.834475 13955 autolog.h:53] enable_memory_optim: True
I1125 08:10:45.834483 13955 autolog.h:54] enable_tensorrt: 0
I1125 08:10:45.834518 13955 autolog.h:55] enable_mkldnn: False
I1125 08:10:45.834525 13955 autolog.h:56] cpu_math_library_num_threads: 10
I1125 08:10:45.834532 13955 autolog.h:57] ----------------------- Data info -----------------------
I1125 08:10:45.834540 13955 autolog.h:58] batch_size: 1
I1125 08:10:45.834547 13955 autolog.h:59] input_shape: dynamic
I1125 08:10:45.834556 13955 autolog.h:60] data_num: 1
I1125 08:10:45.834564 13955 autolog.h:61] ----------------------- Model info -----------------------
I1125 08:10:45.834573 13955 autolog.h:62] model_name: rec
I1125 08:10:45.834579 13955 autolog.h:63] precision: fp32
I1125 08:10:45.834586 13955 autolog.h:64] ----------------------- Perf info ------------------------
I1125 08:10:45.834594 13955 autolog.h:65] Total time spent(ms): 2739
I1125 08:10:45.834602 13955 autolog.h:67] preprocess_time(ms): 10.6524, inference_time(ms): 1269.55, postprocess_time(ms): 0.009118
```

### 3 注意

* 在使用Paddle预测库时，推荐使用2.1.0版本的预测库。
