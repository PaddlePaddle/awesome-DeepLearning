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

#include <include/video_rec.h>

namespace PaddleVideo
{
    void VideoRecognizer::Run(const std::vector<string> &frames_batch_path, const std::vector<std::vector<cv::Mat> > &frames_batch, std::vector<double> *times)
    {
        // Copy parameters to the function
        int real_batch_num = frames_batch.size();

        std::vector<cv::Mat> srcframes(real_batch_num * this->num_seg, cv::Mat());

        for (int i = 0; i < real_batch_num; ++i)
        {
            for (int j = 0; j < this->num_seg; ++j)
            {
                frames_batch[i][j].copyTo(srcframes[i * this->num_seg + j]);
            }
        }

        auto preprocess_start = std::chrono::steady_clock::now();
        /* Preprocess */
        std::vector<cv::Mat> resize_frames;
        std::vector<cv::Mat> crop_frames;
        std::vector<float> input;
        int num_views = 1;

        if (this->inference_model_name == "ppTSM")
        {
            num_views = 1;
            // 1. Scale
            resize_frames = std::vector<cv::Mat>(real_batch_num * this->num_seg, cv::Mat());
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->scale_op_.Run(srcframes[i * this->num_seg + j], resize_frames[i * this->num_seg + j], this->use_tensorrt_, 256);
                }
            }

            // 2. CenterCrop
            crop_frames = std::vector<cv::Mat>(real_batch_num * num_views * this->num_seg, cv::Mat());
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->centercrop_op_.Run(resize_frames[i * this->num_seg + j], crop_frames[i * this->num_seg + j], this->use_tensorrt_, 224);
                }
            }

            // 3. Normalization(inplace operation)
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    for (int k = 0; k < num_views; ++k)
                    {
                        this->normalize_op_.Run(&crop_frames[i * num_views * this->num_seg + j * num_views + k], this->mean_, this->scale_, this->is_scale_);
                    }
                }
            }

            // 4. Image2Array
            int rh = crop_frames[0].rows;
            int rw = crop_frames[0].cols;
            int rc = crop_frames[0].channels();
            input = std::vector<float>(real_batch_num * num_views * this->num_seg *  crop_frames[0].rows * crop_frames[0].cols * rc, 0.0f);
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    for (int k = 0; k < num_views; ++k)
                    {
                        this->permute_op_.Run(&crop_frames[i * num_views * this->num_seg + j * num_views + k], input.data() + (i * num_views * this->num_seg + j * num_views + k) * (rh * rw * rc));
                    }
                }
            }
        }
        else if(this->inference_model_name == "ppTSN")
        {
            num_views = 10;
            // 1. Scale
            resize_frames = std::vector<cv::Mat>(real_batch_num * this->num_seg, cv::Mat());
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->scale_op_.Run(srcframes[i * this->num_seg + j], resize_frames[i * this->num_seg + j], this->use_tensorrt_, 256);
                }
            }

            // 2. TenCrop
            crop_frames = std::vector<cv::Mat>(real_batch_num * this->num_seg * num_views, cv::Mat());
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->tencrop_op_.Run(resize_frames[i * this->num_seg + j], crop_frames, (i * this->num_seg  + j) * num_views, this->use_tensorrt_, 224);
                }
            }

            // 3. Normalization(inplace operation)
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    for (int k = 0; k < num_views; ++k)
                    {
                        this->normalize_op_.Run(&crop_frames[i * this->num_seg * num_views + j * num_views + k], this->mean_, this->scale_, this->is_scale_);
                    }
                }
            }

            // 4. Image2Array
            int rh = crop_frames[0].rows;
            int rw = crop_frames[0].cols;
            int rc = crop_frames[0].channels();
            input = std::vector<float>(real_batch_num * this->num_seg * num_views *  crop_frames[0].rows * crop_frames[0].cols * rc, 0.0f);
            for (int i = 0; i < real_batch_num; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    for (int k = 0; k < num_views; ++k)
                    {
                        this->permute_op_.Run(&crop_frames[i * this->num_seg * num_views + j * num_views + k], input.data() + (i * this->num_seg * num_views + j * num_views + k) * (rh * rw * rc));
                    }
                }
            }
        }
        else
        {
            throw "[Error] Not implemented yet";
        }
        auto preprocess_end = std::chrono::steady_clock::now();

        /* Inference */
        auto input_names = this->predictor_->GetInputNames();
        auto input_t = this->predictor_->GetInputHandle(input_names[0]);
        input_t->Reshape({real_batch_num * num_views * this->num_seg, 3, crop_frames[0].rows, crop_frames[0].cols});
        auto inference_start = std::chrono::steady_clock::now();
        input_t->CopyFromCpu(input.data());
        this->predictor_->Run(); // Use the inference library to predict

        std::vector<float> predict_batch;
        auto output_names = this->predictor_->GetOutputNames();
        auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
        auto predict_shape = output_t->shape();

        // Get the number of class
        int class_num = predict_shape[1];

        int out_numel = std::accumulate(predict_shape.begin(), predict_shape.end(), 1, std::multiplies<int>());
        predict_batch.resize(out_numel); // NxC
        output_t->CopyToCpu(predict_batch.data()); // Copy the model output to predict_batch

        // Convert output (logits) into probabilities
        for (int i = 0; i < real_batch_num; ++i)
        {
            this->softmax_op_.Inplace_Run(predict_batch.begin() + i * class_num, predict_batch.begin() + (i + 1) * class_num);
        }

        auto inference_end = std::chrono::steady_clock::now();

        // output decode
        auto postprocess_start = std::chrono::steady_clock::now();
        std::vector<std::string> str_res;
        std::vector<float>scores;

        for (int i = 0; i < real_batch_num; ++i)
        {
            int argmax_idx = int(Utility::argmax(predict_batch.begin() + i * class_num, predict_batch.begin() + (i + 1) * class_num));
            float score = predict_batch[argmax_idx];
            scores.push_back(score);
            str_res.push_back(this->label_list_[argmax_idx]);
        }
        auto postprocess_end = std::chrono::steady_clock::now();
        for (int i = 0; i < str_res.size(); i++)
        {
            std::cout << frames_batch_path[i] << "\tclass: " << str_res[i] << "\tscore: " << scores[i] << endl;
        }

        std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
        times->push_back(double(preprocess_diff.count() * 1000));
        std::chrono::duration<float> inference_diff = inference_end - inference_start;
        times->push_back(double(inference_diff.count() * 1000));
        std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
        times->push_back(double(postprocess_diff.count() * 1000));
    }

    void VideoRecognizer::LoadModel(const std::string &model_dir)
    {
        //   AnalysisConfig config;
        paddle_infer::Config config;
        config.SetModel(model_dir + "/" + this->inference_model_name + ".pdmodel",
                        model_dir + "/" + this->inference_model_name + ".pdiparams");

        if (this->use_gpu_)
        {
            config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
            if (this->use_tensorrt_)
            {
                auto precision = paddle_infer::Config::Precision::kFloat32;
                if (this->precision_ == "fp16")
                {
                    precision = paddle_infer::Config::Precision::kHalf;
                }
                else if (this->precision_ == "int8")
                {
                    precision = paddle_infer::Config::Precision::kInt8;
                }

                if (this->inference_model_name == "ppTSM" || this->inference_model_name == "TSM")
                {
                    config.EnableTensorRtEngine(
                        1 << 20, this->rec_batch_num * this->num_seg * 1, 3,
                        precision,
                        false, false);
                }
                else if(this->inference_model_name == "ppTSN" || this->inference_model_name == "TSN")
                {
                    config.EnableTensorRtEngine(
                        1 << 20, this->rec_batch_num * this->num_seg * 10, 3,
                        precision,
                        false, false);
                }
                else
                {
                    config.EnableTensorRtEngine(
                        1 << 20, this->rec_batch_num, 3,
                        precision,
                        false, false);
                }
                // std::map<std::string, std::vector<int>> min_input_shape =
                // {
                //     {"data_batch", {1, 1, 1, 1, 1}}
                // };
                // std::map<std::string, std::vector<int>> max_input_shape =
                // {
                //     {"data_batch", {10,  this->num_seg, 3, 224, 224}}
                // };
                // std::map<std::string, std::vector<int>> opt_input_shape =
                // {
                //     {"data_batch", {this->rec_batch_num,  this->num_seg, 3, 224, 224}}
                // };

                // config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                //                               opt_input_shape);
            }
        }
        else
        {
            config.DisableGpu();
            if (this->use_mkldnn_)
            {
                config.EnableMKLDNN();
                // cache 10 different shapes for mkldnn to avoid memory leak
                config.SetMkldnnCacheCapacity(10);
            }
            config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
        }

        config.SwitchUseFeedFetchOps(false);
        // true for multiple input
        config.SwitchSpecifyInputNames(true);

        config.SwitchIrOptim(true);

        config.EnableMemoryOptim();
        config.DisableGlogInfo();

        this->predictor_ = CreatePredictor(config);
    }

} // namespace PaddleVideo
