English | [简体中文](./readme.md)

# Server-side C++ prediction

This chapter introduces the C++ deployment method of the PaddleVideo model. For the python prediction deployment method, please refer to the **Model Reasoning** chapter of the respective model.
C++ is better than python in terms of performance calculation. Therefore, in most CPU and GPU deployment scenarios, C++ deployment methods are mostly used. This section will introduce how to configure the C++ environment in the Linux (CPU/GPU) environment and complete it.
PaddleVideo model deployment.

## 1. Prepare the environment

- For Linux environment, docker is recommended.

- Windows environment, currently supports compilation based on `Visual Studio 2019 Community` (TODO)

* This document mainly introduces the PaddleVideo C++ prediction process based on the Linux environment. If you need to perform C++ prediction based on the prediction library under Windows, please refer to [Windows Compilation Tutorial](./docs/windows_vs2019_build.md)(TODO) for the specific compilation method
* **The purpose of preparing the environment is to get the compiled opencv library and paddle prediction library**.

### 1.1 Compile opencv library

* First, you need to download the compressed package compiled from the source code in the Linux environment from the opencv official website, and unzip it into a folder. Take opencv3.4.7 as an example, the download command is as follows:

    ```bash
    cd deploy/cpp_infer
    wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
    tar -xf 3.4.7.tar.gz
    ```

    After decompression, you can get the decompressed folder of `opencv-3.4.7` in the `deploy/cpp_infer` directory.

* Install ffmpeg

    Opencv and ffmpeg can read the video normally under linux, otherwise it may encounter the situation that the number of video frames returns to 0 or no video frame can be read

    Using a relatively simple apt installation, the installation command is as follows:

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

* To prepare to compile opencv, first enter the `opencv-3.4.7` folder, and then set the opencv source path `root_path` and the installation path `install_path`. The execution command is as follows:

    ```bash
    cd opencv-3.4.7

    root_path=/xxx/xxx/xxx/xxx/opencv-3.4.7 # That is the absolute path of opencv-3.4.7
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

    After the completion of `make install`, opencv header files and library files will be generated in this folder, which will be used to compile the Video inference C++ code later.

    Finally, the installation path `install_path` will be used as the specified path, and a folder of `opencv3` will be obtained. The file structure is shown below.

    ```
    opencv3/
    ├── bin/
    ├── include/
    ├── lib/
    ├── lib64/
    └── share/
    ```

### 1.2 Download or compile Paddle prediction library

There are two ways to obtain the Paddle prediction library, which will be described in detail below.


#### 1.2.1 Download and install directly

* [Paddle prediction library official website](https://paddleinference.paddlepaddle.org.cn/v2.1/user_guides/download_lib.html) provides different cuda versions of Linux prediction libraries, you can Check and **select the appropriate prediction library version** on the official website (it is recommended to select the prediction library with paddle version>=2.0.1).

* Download and get a `paddle_inference.tgz` compressed package, and then unzip it into a folder, the command is as follows (taking the machine environment as gcc8.2 as an example):

    ```bash
    wget https://paddle-inference-lib.bj.bcebos.com/2.1.1-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddle_inference.tgz
    tar -xf paddle_inference.tgz
    ```

    Eventually, a subfolder of `paddle_inference/` will be generated in the current folder.

#### 1.2.2 Prediction library source code compilation
* If you want to get the latest prediction library features, you can clone the latest code from Paddle github and compile the prediction library from source code.
* You can refer to [Paddle prediction library installation and compilation instructions](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) instructions from github Obtain the Paddle code, and then compile it to generate the latest prediction library. The method of using git to get the code is as follows.

    ```shell
    git clone https://github.com/PaddlePaddle/Paddle.git
    git checkout release/2.1
    ```

* After entering the Paddle directory, the compilation method is as follows.

    ```shell
    rm -rf build
    mkdir build
    cd build

    cmake .. \
        -DWITH_CONTRIB=OFF \
        -DWITH_MKL=ON \
        -DWITH_MKLDNN=ON \
        -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_INFERENCE_API_TEST=OFF \
        -DON_INFER=ON \
        -DWITH_PYTHON=ON
    make -j
    make inference_lib_dist -j4 # 4为编译时使用核数，可根据机器情况自行修改
    ```

    You can refer to [documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#congyuanmabianyi) for more introduction of compilation parameter options.


* After the compilation is complete, you can see the following files and folders are generated under the file `build/paddle_inference_install_dir/`.

    ```bash
    build/paddle_inference_install_dir/
    ├── CMakeCache.txt
    ├── paddle/
    ├── third_party
    └── version.txt
    ```

    Among them, `paddle` is the Paddle library required for C++ prediction, and `version.txt` contains the version information of the current prediction library.

## 2. Compile and run the prediction demo

### 2.1 Export the model as an inference model

* This step is the same as the export prediction model under the python deployment mode. You can refer to the model prediction chapter of the respective model. Several related inference model files exported are used for model prediction. **Taking PP-TSM as an example**, the directory structure of the derived prediction model is as follows.

    ```
    inference/
    └── ppTSM/
    ├── ppTSM.pdiparams
    ├── ppTSM.pdiparamsinfo
    └── ppTSM.pdmodel
    ```


### 2.2 Compile PaddleVideo C++ prediction demo

* Enter the `deploy/cpp_infer` directory and execute the following compile command

    ```shell
    bash tools/build.sh
    ```

    The addresses of the Paddle C++ prediction library, opencv and other dependent libraries in `tools/build.sh` need to be replaced with the actual addresses on your own machine.

* Specifically, you need to modify the environment path in `tools/build.sh`, the relevant content is as follows:

    ```shell
    OPENCV_DIR=your_opencv_dir
    LIB_DIR=your_paddle_inference_dir
    CUDA_LIB_DIR=your_cuda_lib_dir
    CUDNN_LIB_DIR=your_cudnn_lib_dir
    TENSORRT_DIR=your_tensorRT_dir
    ```

    Take PP-TSM as an example, the above parameters are as follows (the xxx part is modified according to the user's own machine situation)

    ```bash
    OPENCV_DIR=/xxx/xxx/xxx/xxx/xxx/xxx/opencv3
    LIB_DIR=/xxx/xxx/xxx/xxx/xxx/paddle_inference
    CUDA_LIB_DIR=/xxx/xxx/cuda-xxx/lib64
    CUDNN_LIB_DIR=/xxx/xxx/cuda-xxx/lib64
    TENSORRT_DIR=/xxx/xxx/TensorRT-7.0.0.11
    ```

    Among them, `OPENCV_DIR` is the address where opencv is compiled and installed; `LIB_DIR` is the download (`paddle_inference` folder) or compiled Paddle prediction library address (`build/paddle_inference_install_dir` folder); `CUDA_LIB_DIR` is the cuda library file address , In docker, it is `/usr/local/cuda/lib64`; `CUDNN_LIB_DIR` is the address of the cudnn library file, in docker it is `/usr/lib/x86_64-linux-gnu/`. **Note: The above paths are written as absolute paths, do not write relative paths. **


* After the compilation is complete, an executable file named `ppvideo` will be generated in the `cpp_infer/build` folder.


### 2.3 Run PaddleVideo C++ prediction demo

Operation mode:

```bash
./build/ppvideo <mode> [--param1] [--param2] [...]
```

Among them, `mode` is a required parameter, which means the selected function, and the value range is ['rec'], which means **video recognition** (more functions will be added in succession).

##### 1. Call video recognition:
```bash
# run PP-TSM inference
./build/ppvideo rec \
    --rec_model_dir=../../inference/ppTSM \
    --inference_model_name=ppTSM \
    --video_dir=./example_video_dir \
    --num_seg=8 \
    --seg_len=1

# run PP-TSN inference
./build/ppvideo rec \
    --rec_model_dir=../../inference/ppTSN \
    --inference_model_name=ppTSN \
    --video_dir=./example_video_dir \
    --num_seg=25 \
    --seg_len=1
```
More parameters are as follows:

- General parameters

    | Parameter name | Type | Default parameter | Meaning |
    | ------------- | ---- | --------------- | ------------------------------------------------------------ |
    | use_gpu | bool | false | Whether to use GPU |
    | gpu_id | int | 0 | GPU id, valid when using GPU |
    | gpu_mem | int | 4000 | GPU memory requested |
    | cpu_threads | int | 10 | The number of threads for CPU prediction. When the number of machine cores is sufficient, the larger the value, the faster the prediction speed |
    | enable_mkldnn | bool | false | Whether to use mkldnn library |
    | use_tensorrt | bool | false | Whether to use the tensorrt library |
    | precision | str | "fp32" | Use fp32/fp16/uint8 precision to predict |
    | benchmark | bool | true | Whether to enable benchmark during prediction, after enabling it, the configuration, model, time-consuming and other information will be output at the end. |
    | save_log_path | str | "./log_output/" | Prediction result save directory |

- Video recognition model related

    | Parameter name | Type | Default parameter | Meaning |
    | -------------- | ------ | --------------------------------------------- | ------------------------------------ |
    | video_dir | string | "../example_video_dir" | The path of the folder where the video to be recognized is stored |
    | rec_model_dir | string | "" | The folder path where the exported prediction model is stored |
    | inference_model_name | string | "ppTSM" | The name of the model used in the prediction |
    | num_seg | int | 8 | Number of video segments |
    | seg_len | int | 1 | The number of frames extracted in each segment of the video |
    | rec_batch_num | int | 1 | Batch size during model prediction |
    | char_list_file | str | "../../data/k400/Kinetics-400_label_list.txt" | The text path for storing all category labels and corresponding names |

​	Take the sample video `example01.avi` under example_video_dir as the input video as an example, the final 	screen will output the detection results as follows.

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

### 3 Attention

* When using the Paddle prediction library, it is recommended to use the prediction library of version 2.1.0.
