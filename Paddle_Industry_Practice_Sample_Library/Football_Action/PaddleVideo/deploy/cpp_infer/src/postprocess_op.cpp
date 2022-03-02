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

#include <include/postprocess_op.h>

namespace PaddleVideo
{
    void Softmax::Inplace_Run(const std::vector<float>::iterator &_begin, const std::vector<float>::iterator &_end)
    {
        const float max_value = *std::max_element(_begin, _end);
        float denominator = 0.0f;
        for (auto it = _begin; it != _end; ++it)
        {
            *it = std::exp((*it) - max_value);
            denominator += (*it);
        }
        for (auto it = _begin; it != _end; ++it)
        {
            *it /= denominator;
        }
    }
    std::vector<float> Softmax::Run(const std::vector<float>::iterator &_begin, const std::vector<float>::iterator &_end)
    {
        std::vector<float> prob(_begin, _end);
        const float max_value = *std::max_element(prob.begin(), prob.end());
        float denominator = 0.0f;
        for (auto it = _begin, it_p = prob.begin(); it != _end; ++it, ++it_p)
        {
            (*it_p) = std::exp((*it) - max_value);
            denominator += (*it_p);
        }
        for (auto it = prob.begin(); it != prob.end(); ++it)
        {
            (*it) /= denominator;
        }
        return prob;
    }

} // namespace PaddleVideo
