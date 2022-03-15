// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"
#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/video_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include "auto_log/autolog.h"

// general parameters
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8.");
DEFINE_bool(benchmark, true, "Whether use benchmark.");
DEFINE_string(save_log_path, "./log_output/", "Save benchmark log path.");


// video recognition related
DEFINE_string(video_dir, "", "Dir of input video(s).");
DEFINE_string(rec_model_dir, "../example_video_dir", "Path of video rec inference model.");
DEFINE_string(inference_model_name, "ppTSM", "The name of the model used in the prediction.");
DEFINE_int32(num_seg, 8, "number of frames input to model, which are extracted from a video.");
DEFINE_int32(seg_len, 1, "number of frames from a segment.");
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
DEFINE_string(char_list_file, "../../data/k400/Kinetics-400_label_list.txt", "Path of dictionary.");


using namespace std;
using namespace cv;
using namespace PaddleVideo;


static bool PathExists(const std::string& path)
{
#ifdef _WIN32
    struct _stat buffer;
    return (_stat(path.c_str(), &buffer) == 0);
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}


int main_rec(std::vector<cv::String> &cv_all_video_names)
{
    std::vector<double> time_info = {0, 0, 0}; // Statement time statistics vector
    VideoRecognizer rec(FLAGS_rec_model_dir, FLAGS_inference_model_name, FLAGS_use_gpu, FLAGS_num_seg,
                        FLAGS_rec_batch_num, FLAGS_gpu_id,
                        FLAGS_gpu_mem, FLAGS_cpu_threads,
                        FLAGS_enable_mkldnn, FLAGS_char_list_file,
                        FLAGS_use_tensorrt, FLAGS_precision); // Instantiate a video recognition object

    int batch_num = FLAGS_rec_batch_num;
    for (int i = 0, n = cv_all_video_names.size(); i < n; i += batch_num) // Process each video
    {
        int start_idx = i;
        int end_idx = min(i + batch_num, n);
        std::vector<std::vector<cv::Mat> > frames_batch;
        for (int j = start_idx; j < end_idx; ++j)
        {
            std::vector<cv::Mat> frames = Utility::SampleFramesFromVideo(cv_all_video_names[i], FLAGS_num_seg, FLAGS_seg_len);
            frames_batch.emplace_back(frames);
        }
        std::vector<double> rec_times; // Initialization time consumption statistics

        // Take the read several video frames and send them to the run method of the recognition class to predict
        rec.Run(std::vector<string>(cv_all_video_names.begin() + start_idx, cv_all_video_names.begin() + end_idx), frames_batch, &rec_times);

        time_info[0] += rec_times[0];
        time_info[1] += rec_times[1];
        time_info[2] += rec_times[2];
    }
    if (FLAGS_benchmark)
    {
        AutoLogger autolog("rec",
                           FLAGS_use_gpu,
                           FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn,
                           FLAGS_cpu_threads,
                           FLAGS_rec_batch_num,
                           "dynamic",
                           FLAGS_precision,
                           time_info,
                           cv_all_video_names.size()); // Generate detailed information on the run
        autolog.report(); // Print running details
    }

    return 0;
}


void check_params(char* mode)
{
    if (strcmp(mode, "rec") == 0)
    {
        std::cout << "[" << FLAGS_rec_model_dir << "]" << std::endl;
        std::cout << "[" << FLAGS_video_dir << "]" << std::endl;
        if (FLAGS_rec_model_dir.empty() || FLAGS_video_dir.empty())
        {
            std::cout << "Usage[rec]: ./ppvideo --rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                      << "--video_dir=/PATH/TO/INPUT/VIDEO/" << std::endl;
            exit(1);
        }
    }
    if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" && FLAGS_precision != "int8")
    {
        cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. " << endl;
        exit(1);
    }
}


int main(int argc, char **argv)
{
    if (argc <= 1 || (strcmp(argv[1], "rec") != 0)) //Get user input and check
    {
        std::cout << "Please choose one mode of [rec] !" << std::endl;
        return -1;
    }
    std::cout << "mode: " << argv[1] << endl; // Type of inference task required for output

    // Parsing command-line
    google::ParseCommandLineFlags(&argc, &argv, true);
    check_params(argv[1]);

    if (!PathExists(FLAGS_video_dir)) // Determine whether the directory where the video exists
    {
        std::cerr << "[ERROR] video path not exist! video_dir: " << FLAGS_video_dir << endl;
        exit(1);
    }

    std::vector<cv::String> cv_all_video_names; // Store all video paths

    cv::glob(FLAGS_video_dir, cv_all_video_names); // Search all videos under FLAGS_video_dir, save in cv_all_video_names
    std::cout << "total videos num: " << cv_all_video_names.size() << endl; // 输出搜索到的视频个数

    if (strcmp(argv[1], "rec") == 0)
    {
        return main_rec(cv_all_video_names); // Output the number of videos searched
    }
    return 0;
}
