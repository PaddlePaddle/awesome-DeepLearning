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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/preprocess_op.h>


namespace PaddleVideo
{

    void Permute::Run(const cv::Mat *im, float *data)
    {
        int rh = im->rows;
        int rw = im->cols;
        int rc = im->channels();
        for (int i = 0; i < rc; ++i)
        {
            // Extract the i-th channel of im and write it into the array with (data + i * rh * rw) as the starting address
            cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), rc - 1 - i);
        }
    }

    void Normalize::Run(cv::Mat *im, const std::vector<float> &mean,
                        const std::vector<float> &scale, const bool is_scale)
    {
        double e = 1.0;
        if (is_scale)
        {
            e /= 255.0;
        }
        (*im).convertTo(*im, CV_32FC3, e);
        std::vector<cv::Mat> bgr_channels(3);
        cv::split(*im, bgr_channels);
        for (auto i = 0; i < bgr_channels.size(); i++)
        {
            bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 / scale[i], (0.0 - mean[i]) / scale[i]);
        }
        cv::merge(bgr_channels, *im);
    }

    void Scale::Run(const cv::Mat &img, cv::Mat &resize_img, bool use_tensorrt, const int &short_size)
    {
        int h = img.rows;
        int w = img.cols;
        if ((w <= h && w == short_size) || (h <= w && h == short_size))
        {
            img.copyTo(resize_img);
        }
        else
        {
            int oh, ow;
            if (w < h)
            {
                ow = short_size;
                oh = h * ow / w;
            }
            else
            {
                oh = short_size;
                ow = w * oh / h;
            }
            cv::resize(img, resize_img, cv::Size(ow, oh), 0.0f, 0.0f, cv::INTER_LINEAR);
        }
    }

    void CenterCrop::Run(const cv::Mat &img, cv::Mat &crop_img, bool use_tensorrt, const int &target_size)
    {
        int h = img.rows;
        int w = img.cols;
        int crop_h = target_size;
        int crop_w = target_size;
        if (w < crop_w || h < crop_h)
        {
            printf("[Error] image width (%d) and height (%d) should be larger than crop size (%d)",
                   w, h, target_size);
        }
        else
        {
            int x1 = (w - crop_w) / 2;
            int y1 = (h - crop_h) / 2;
            crop_img = img(cv::Rect(x1, y1, crop_w, crop_h));
        }
    }

    void TenCrop::Run(const cv::Mat &img, std::vector<cv::Mat> &crop_imgs, const int &begin_index, bool use_tensorrt, const int &target_size)
    {
        int h = img.rows;
        int w = img.cols;
        int crop_h = target_size;
        int crop_w = target_size;
        int w_step = (w - crop_w) / 4;
        int h_step = (h - crop_h) / 4;
        pair<int, int>offsets[5] =
        {
            {0,          0},
            {4 * w_step, 0},
            {0,          4 * h_step},
            {4 * w_step, 4 * h_step},
            {2 * w_step, 2 * h_step}
        };
        for (int i = 0; i < 5; ++i)
        {
            const int &j = i * 2;
            const int &x1 = offsets[i].first;
            const int &y1 = offsets[i].second;
            crop_imgs[begin_index + j] = img(cv::Rect(x1, y1, crop_w, crop_h)); // cropped
            cv::flip(img(cv::Rect(x1, y1, crop_w, crop_h)), crop_imgs[begin_index + j + 1], 0); // cropped
        }
    }
} // namespace PaddleVideo
